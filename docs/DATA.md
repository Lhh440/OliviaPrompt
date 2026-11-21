# 数据管线与训练（M4）

> 核心认知：公开数据集只有「输出侧」（成品提示词），没有「输入侧」（用户原话需求）。
> 输入侧必须自己造——这份文档描述的就是造法、练法、验法。

---

## 一、数据从哪来（合成，不爬取）

```
M1 采样器（scenes.json × activities.json × pools.py）
   │  一个「已决定的完整世界」= 44 槽位的 payload（src 全部可追溯）
   ▼
中文意图模板（synthesize.make_intent，三档口吻 × 句式变体）
   │  · 短 ≤15 字：「银发女子坐在老式书房里画画」
   │  · 中 ~60 字：+ 情绪 + 光线方向 + 机位 + 留白
   │  · 长 ~150 字：+ 叙事 + 显式约束（脸手可见 / 背景简洁 / 无文字水印）
   ▼
SFT 对：input = "[档位:tier] 中文需求"   output = 该档位完整槽位 JSON
```

**关键决定：训练目标是槽位 JSON，不是散文。**

1. JSON 输出比 500 词散文短一半，`cutoff_len: 1536` 够用，CPU 推理也快一倍
2. 格式永远由代码（渲染器）保证，模型只学「决策」
3. 散文的风格是渲染器的事——改模板/fork 骨架不需要重训

**src 标记是监督信号的一部分。** 意图里「说出口的信息」→ `user`；
采样器推出来的 → `inferred`；没动的 → `default`。模型由此学会区分
「用户要的」和「我猜的」，这正是产品里「不追问用户」机制的前提。

## 二、去重与覆盖（方案 §六 硬规则）

| 规则 | 实现 |
|---|---|
| 同一「场景 × 动作」组合 ≤ 3 次 | `sample.MAX_SCENE_ACTIVITY_USES`，超出即拒绝 |
| 同一底图出 2–3 个措辞变体 | 每档位 2–3 种句式（`make_intent` 的 shape 分支） |
| 训前跑槽位覆盖矩阵 | `olivia-prompt coverage` / `synthesize` 的 `coverage.txt` |
| 任一开放槽 < 15 种独立取值 → 补数据 | 只对**开放词汇槽**告警；封闭域（媒介/机位/画幅等取值空间本来就小）只报告不告警 |

物理诚实性约束（防止采样出「走着画画」）：
`walk` 只配 `{map, photograph, smoke}`；`lie` 只配 `{read}`（见 `sample.WALK_OK/LIE_OK`）。

规模目标（去重后）：短 1500 / 中 2000 / 长 2000 / 难例（双主体）500。
一条基准记录 × 3 档位 = 3 条 SFT 对，`--n 2000` 即达成。

**覆盖门禁现状（n=200 验收跑，见 `olivia-prompt coverage`）：**
setting 58 种 / outfit_main 24 / hair_color 16 / craft 组合 100 / perch 155 —— 健康；
仍低于 15 的槽位（skin 6、grip 8、outfit_layer 10 等）是**池子上限**，
合成全量（n=2000）会自然翻倍；要彻底过线就往 `pools.py` 加条目，一行一个。

## 三、训练（LLaMA-Factory，Windows 优先）

```
training/
├─ data/synthesize.py        合成器（含 --base-url 可选强模型改写口吻）
├─ configs/lite-4b.yaml      Qwen3-4B  · 4070 ≈ 25 min
├─ configs/standard-8b.yaml  Qwen3-8B  · 4070 ≈ 55 min
├─ scripts/train.bat         一键：ComfyUI 检查 → 合成 → venv+LF → 训练 → 导出
├─ scripts/export_gguf.py    merge LoRA → convert_hf_to_gguf → Q4_K_M/Q5_K_M
├─ scripts/eval.py           50 条基准评测 + 基座对比 + 可选 round-trip
└─ eval/benchmark.jsonl      50 条基准（已入库，评测可比性的锚）
```

```bat
:: 全流程
training\scripts\train.bat lite
training\scripts\train.bat standard
:: 分步
training\scripts\train.bat lite data     :: 只合成数据
training\scripts\train.bat lite train    :: 只训练
training\scripts\train.bat lite export   :: 只导出 GGUF
```

统一配置即方案 §七：LoRA r32 α64 / HQQ 4bit QLoRA / lr 8e-5 cosine / 3 epochs /
`cutoff_len 1536` / sdpa（FA2 无 Windows 官方 wheel）/ neftune 5。
`template: qwen3`，老版本 LLaMA-Factory 没有该模板时改回 `qwen2`。

训练产物不入库（`training/data/out/`、`training/out/`、`.venv-train/` 已 gitignore）；
GGUF 按方案 §八放 Release + `models.json` 清单（M5）。

## 四、评测（数据说话，不吹）

`training/scripts/eval.py` 对 50 条基准测四个维度：

| 指标 | 含义 |
|---|---|
| `json_ok_rate` | 模型是否输出可解析的槽位 JSON（格式生死线） |
| `slot_acc` | 期望槽位命中（exact / contains 两级） |
| `weighted_acc` | 按影响权重加权：高×3 / 中×2 / 低×1 |
| `consistency_error_rate` | 自洽校验 error 数（坐姿没地面、机位矛盾…） |

**方案 §十 强制实验——「这跟直接用 Qwen 有什么区别」：**

```bash
# 微调版（用训练时的紧凑指令）
python training/scripts/eval.py --base-url http://127.0.0.1:8081/v1 --model olivia --label lite
# 原始基座（用产品版 few-shot 系统提示，给它最好的机会）
python training/scripts/eval.py --base-url <基座> --model Qwen3-4B --baseline --label base
# 对比
python training/scripts/eval.py --compare training/eval/report_lite.json training/eval/report_base.json
```

`--roundtrip` 可选：对前若干条做 round-trip 质检（强模型判断「槽位表 vs 需求」
的 missing/extra），对应产品的 round-trip 自检特性。

## 五、已知边界（诚实清单）

* **无人物纯场景不支持**：`subject_count` 枚举 1–3，0 未实现（渲染器 P1/P2 需整体跳过，改动大）。难例以双主体替代。
* **双主体的散文是复数化的单句**：代词/动词已复数（they wear / their），但衣饰从句仍是单人语法，基准里作为难例单独统计。
* **意图模板是规则句式**：多样性来自池子×句式×档位；要更自然的口吻，跑
  `synthesize.py --base-url <强模型端点>` 让强模型改写（改写失败自动回落模板）。
* **骨架变体 B/C/D 不进训练目标**：模型输出槽位 JSON，骨架是渲染期选择
  （`render --skeleton B`），所以换骨架零成本、且永远不会被模型「写坏」。
  变体 B/C/D 尚需作者过目确认口吻（方案 §十一.2）。
