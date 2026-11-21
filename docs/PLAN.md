# OliviaPrompt · 方案 v0.1

> 本地优先、自带模型的 AI 绘画提示词生成器。
> 一句话定位：**把你脑子里的画面，变成写对的正向 + 负向提示词。不需要懂英文，不需要调 API。**

---

## 一、竞品调研结论

我在 GitHub 上把同类项目扫了一遍（按 star 排序，截止 2026-11）。结论是：**没有人在做"本地自带小模型 + 中文需求理解 + 三档长度 + 正反向成套"这个组合**。

| 项目 | star | 许可 | 它是什么 | 缺什么 |
|---|---|---|---|---|
| `wfjsw/danbooru-diffusion-prompt-builder` | 1262 | AGPL-3.0 | Danbooru/NAI 标签超市，手动点选 | 不解需求，无模型 |
| `SuzumiyaAkizuki/DanbooruSearchOnline` | 301 | GPL-3.0 | 语义匹配：日常语言 → 标准 tag | 纯检索，不做意图推理，不出散文，无负向 |
| `p1atdev/sd-danbooru-tags-upsampler` | 108 | Apache-2.0 | 小模型把 tag 串补详细 | 只在 tag 空间里扩，不出自然语言 |
| `Mexes-GM/booru-prompt-gallery` | 33 | AGPL-3.0 | 从 booru 选图 → 复制干净 prompt（Illustrious/Pony/SDXL/Anima） | 从"已有图"出发，不从"需求"出发 |
| `yu20000505/prompt-forge` | 23 | MIT | **一句话 → NAI 正向 tag 流 + UC 负向，本地跑提示词体检，纯本地零依赖** | **直连 OpenAI 兼容接口，要用户自备模型** |
| `1756141021/tag-skill` | 24 | MIT | Agent skill 形态的 tag 生成 | 依赖宿主 agent 的模型 |

### 我们的差异点（就一句）

`prompt-forge` 是最接近的对手，产品形态几乎一样——但它要求用户**自己有一个 OpenAI 兼容接口**。也就是说它是个"提示词编排壳"，能力上限取决于用户接的模型。

**OliviaPrompt 把模型也一起发出去。** 用户下载、双击 `start.bat`、浏览器打开，就能用。这是它和所有现存项目的分界线，也是开源分发时唯一的硬门槛（所以模型必须小）。

### 额外的差异化（竞品都没有）

1. **三档长度成套**：短（danbooru tag 串）/ 中（三段散文）/ 长（四段 14 段位），同一份槽位决策，三个渲染器
2. **槽位可见可改**：模型出的不是最终字符串，是 20 个槽位的 JSON。用户改一个槽，即时重渲染，不重跑模型。竞品全是"输入框进、文本出"的黑盒
3. **负向有效性标注**：Illustrious 标"强"，Krea2 Turbo/Flux 标"无效并自动转成正向约束补丁"
4. **round-trip 自检**：生成的 prompt 反喂给 VLM 描述画面，和用户原需求比对，自动给出"漏了什么/多了什么"

---

## 二、仓库结构

```
OliviaPrompt/
├─ apps/
│  ├─ web/                 # Vite + React，复用 OliviaComfyUi/packages/design 的设计语言
│  └─ cli/                 # olivia-prompt "赛博朋克雨夜巷子抽烟的银发女" --tier long
├─ services/
│  └─ core/                # FastAPI + llama.cpp 推理进程（Python）
├─ packages/
│  ├─ schema/              # 槽位 JSON Schema ← 唯一真源
│  ├─ renderer/            # 三档渲染器（Python + TS 双实现，输出必须逐字节一致）
│  ├─ negative/            # 负向 5 层生成器
│  └─ taxonomy/            # 由 Mexes taxonomy 派生的槽位映射表（生成脚本，不入库）
├─ training/
│  ├─ data/                # 数据合成器（只存脚本，不存数据）
│  ├─ configs/             # lite-4b.yaml / standard-8b.yaml
│  ├─ scripts/             # train.bat / export_gguf.py / eval.py
│  └─ eval/                # 50 条 benchmark + round-trip 评测
├─ models/
│  └─ models.json          # 权重清单（URL + sha256），不入库
├─ scripts/
│  ├─ start.bat            # 一键启动
│  ├─ fetch-data.bat       # 拉词库（不打包进仓库）
│  └─ docker-compose.yml
└─ docs/
   ├─ PLAN.md
   ├─ SCHEMA.md
   ├─ NEGATIVE.md
   └─ DATA.md
```

**关键决定：词库不入库。** `fetch-data.bat` 运行时从 HuggingFace 拉，本地缓存。理由是上游 `anhao/ai-pose-gallery-zh` 的许可证是 `NOASSERTION`，替别人承担再分发责任不划算。拉的是：

- `Mexes/danbooru-tag-taxonomy`（CC-BY-4.0，112,570 tag / 7 类 33 槽）
- `ame-la/danbooru-tags-data-zh`（中文译名）
- `gaodrew/midjourney-prompts-highquality`（Apache-2.0，只用于抽短语池）

---

## 三、架构分层

```
中文需求
  │
  ├─[模型]─▶ 槽位 JSON  (20 槽 × 档位子集 + src 标记 + 3 个方向变体)
  │            │
  │            ├─[代码] 自洽校验（规则，坐姿↔双脚着地 / 室内↔无树叶细脉 / 光向↔受光面）
  │            ├─[代码] 渲染器 ×3  ──▶ 短 tag 串 / 中三段 / 长四段
  │            ├─[代码] 负向 5 层  ──▶ 反向提示词
  │            └─[代码] 生态适配    ──▶ Illustrious(强) / SDXL(强) / Turbo(转正向补丁) / MJ(--no)
  │
  └─▶ Web：槽位表可编辑 ──▶ 改一个槽 ──▶ 只重渲染，不重跑模型
```

**核心原则：模型只负责"决策"，所有"格式"都由代码保证。** 这是三档长度几乎零成本、且格式永不崩的唯一做法；也是用户能改模板、fork 自己风格的前提。

---

## 四、槽位 Schema（三档共享超集）

一份 JSON Schema，通过 `tier` 字段裁出子集。槽位来自 `Mexes/danbooru-tag-taxonomy` 的 33 slot 体系（tag 侧的语义是"在 prompt 里会互相替换的 tag 共享一个 slot"，正好等价于我们的槽位定义）。

每个槽位三态来源标记，**不用浮点置信度**（模型自报的 confidence 是幻觉，不可校验）：

| src | 含义 | 行为 |
|---|---|---|
| `user` | 用户明确说了 | 直接用 |
| `inferred` | 模型合理推断 | 用，在摘要里让用户看见 |
| `default` | 无任何信息，取骨架默认 | 高影响槽则标黄提示 |

```
高影响（缺了会崩）：subject_count / gender / pose_family / setting / light_direction / camera_height / aspect / medium
中影响：outfit / props / materials / palette
低影响：render_terms / mood / ambient_detail
```

**短档**（40–80 词，纯 danbooru tag）
`subject_count, solo, hair, eyes, expression, outfit[], pose, camera, background, lighting, style`

**中档**（~150 词，三段散文）
主体与服饰 → 动作与场景 → 光色与渲染

**长档**（300–520 词，四段 14 段位）
前缀行 → 段1 主体/服饰/判词 → 段2 动作/姿态/可见性 → 段3 环境/材质/细节衰减 → 段4 光/色/构图/渲染判词

---

## 五、负向提示词：5 层镜像

不用通用词表堆砌（现代模型对 `worst quality` 基本免疫，堆长了还互相稀释）。从正向逐句反推：

| 层 | 内容 | 规则 |
|---|---|---|
| L0 | 生态判定 | Illustrious/SDXL/NAI 强；Turbo/Flux/Qwen-Image 弱或无效；MJ 转 `--no` |
| L1 | 基底 | 只留 4–6 个：`worst quality, low quality, jpeg artifacts, watermark, text, signature, blurry` |
| L2 | 结构 | 有主体才加：`extra limbs, extra fingers, fused fingers, malformed hands, asymmetrical eyes` |
| L3 | **镜像** | 从正向逐句反推：正向 `holding a pencil` → 负向 `object merging into hand, pencil through fingers`；正向 `a single woman` → `multiple people, duplicate subject` |
| L4 | 风格互斥 | anime → `photorealistic, 3d render`；photo → `anime, cartoon, cel shading` |
| L5 | 长度纪律 | 硬截断 40–60 词，优先级 L1 > L3 > L2 > L4 |

**唯一判断标准**：负向只能排除"错误状态"，不能排除"其他选项"。写了 `silver hair` 就往负向塞 `blonde hair` 会反向污染发色——这是区分负向写得好不好的分水岭。

**生态适配**：负向无效的生态，自动转成正向约束补丁（`不要多余手指` → 正向加 `correct five-fingered hands, natural hand anatomy`）。

---

## 六、数据管线

**公开数据集只有"输出侧"，没有"输入侧"。** 所有 MJ/danbooru 数据都是成品提示词，没有"用户原话需求"。你的产品差异恰恰在输入侧，所以那一半必须自己造。

```
输出侧词库  ← Mexes taxonomy（tag）+ MJ 高质量子集（短语池）+ 你已定的长档骨架
输入侧需求  ← 30–50 场景 × 槽位受约束组合 → 强模型改写成 3 种口吻
                 · 短：一句 15 字「屋顶花园画画的白发女孩」
                 · 中：一段 60 字，含情绪与构图
                 · 长：150 字，含叙事与约束
```

规模目标（去重后）：

| 档位 | 条数 | 来源 |
|---|---|---|
| 短 | 1500 | taxonomy 组合 + 规则渲染 |
| 中 | 2000 | 场景库 × 服饰/姿势 × 改写 |
| 长 | 2000 | 骨架 × 场景 × 多措辞变体 |
| 难例 | 500 | 双人 / 无人物纯场景 / 复杂手部交互 / 极简背景 |

**硬规则**：同一「场景 × 动作」组合最多出现 3 次；同一底图出 2–3 个措辞变体（防背诵）；训前跑**槽位覆盖矩阵**，任一槽独立取值 < 15 种就补数据——这比看 loss 曲线有用。

---

## 七、训练方案（双档）

| | Lite | Standard |
|---|---|---|
| 基座 | Qwen3-4B | Qwen3-8B |
| 导出量化 | Q4_K_M（≈2.5GB） | Q5_K_M（≈5.6GB） |
| 支持档位 | **短 + 中** | 短 + 中 + 长 |
| 目标机器 | 8GB 显卡 / 纯 CPU | 12GB+ |
| 4070 训练耗时 | ≈25 min | ≈55 min |
| 许可证 | Apache-2.0（分发干净） | 同 |

Lite 不支持长档是**刻意的裁剪**：4B 写 500 词四段散文会截断和复读，让它硬写是自欺欺人，文档里明说。

统一训练配置：

```yaml
finetuning_type: lora
quantization_bit: 4
quantization_method: hqq        # Windows 友好，免 CUDA 扩展
lora_rank: 32
lora_alpha: 64
lora_target: all
learning_rate: 8.0e-5
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
cutoff_len: 1536                 # 输出是槽位 JSON，比散文短一半
gradient_checkpointing: true
lr_scheduler_type: cosine
warmup_ratio: 0.05
neftune_noise_alpha: 5
bf16: true
attn_implementation: sdpa        # Windows 上 FA2 无官方 wheel，别碰
packing: false
train_on_prompt: false
```

**模型输出 JSON 而不是散文**，推理速度比出散文快一倍：CPU 上 15–20 秒出结果，这才是可接受的产品体验。

---

## 八、分发

**`start.bat` 为主，docker 为辅。**

```
start.bat
  1. 探测显存（nvidia-smi / dxdiag）
  2. ≥12GB → 提示可下载 Standard；<12GB → 默认 Lite；均可手动覆盖
  3. 检查 models/ → 缺则从 HF 按 models.json 下载（带断点续传 + sha256 校验）
  4. 起 services/core（FastAPI + llama.cpp），端口 7860
  5. 打开浏览器
```

权重不放仓库，放 Release + `models.json` 清单。词库走 `fetch-data.bat`。

---

## 九、里程碑

| 阶段 | 内容 | 估时 | 产出 |
|---|---|---|---|
| M0 | schema + 三档渲染器 + 负向 5 层 + CLI | 2 天 | `olivia-prompt --tier long` 能吃手填 JSON 出完整正反向 |
| M1 | 规则版可用：场景库 × 槽位采样，不跑模型 | 1 天 | 能批量出 200 条看效果，验证骨架覆盖度 |
| M2 | 接入模型（先接 OpenAI 兼容，后接本地 GGUF） | 1 天 | 中文一句话 → 完整输出 |
| M3 | Web 端（槽位表可编辑 + 三档并列 + 负向有效性标注） | 2 天 | 可用产品形态 |
| M4 | 数据合成 + Lite/Standard 两轮训练 + eval | 3 天 | 两个 GGUF + 对比报告 |
| M5 | `start.bat` / docker / 模型上传 / README | 1 天 | 开源发布 |

**M0–M2 必做（4 天），M3 是体验分水岭，M4 才是训练。** 先有闭环再训模型——因为 agent 的质量主要来自 M0 的骨架和 M2 的接入，微调只是让 M2 那一步离线、稳定、省一次调用。

---

## 十、风险与对策

| 风险 | 对策 |
|---|---|
| 上游词库许可证 `NOASSERTION` | 不入库，`fetch-data.bat` 运行时拉取 |
| 4B 长档退化 | 裁剪为短+中，文档明说 |
| "这跟直接用 Qwen 有什么区别" | 必须做 **微调版 vs 原始 4B+few-shot** 的对比评测，数据说话 |
| 长档骨架样本只有 5 条 | 手工抽象 1 主 + 3 变体骨架，变体由你确认 |
| taxonomy 有 11,413 条 `needs_review` | 只用 `approved` 的 101,157 条 |
| 短档 ↔ 长档不是同一映射 | 用 `Mexes/booru-tags-to-natural-language` 搭桥 |
| 训练时 ComfyUI 抢显存 | 训练脚本前置检查，检测到 ComfyUI 进程则提示关闭 |

---

## 十一、实施记录（M0–M4，2026-11）

M0–M4 已完成。逐项对账与两处刻意偏离：

| 阶段 | 交付 | 对账 |
|---|---|---|
| M0 | schema + 三档渲染器 + 负向 5 层 + CLI，33 测试 | ✅ 逐字节确定性有测试锚定 |
| M1 | 场景库 60 场景（人工场景库结构化入 `data/scenes.json`）× 活动池 23 × 池子组合，`sample --n 200` 验收：58 种场景 / 24 套服饰 / 0 自洽错误；覆盖矩阵落地为 `coverage` 命令 | ✅ |
| M2 | OpenAI 兼容客户端（stdlib urllib，零依赖）+ 槽位填充系统提示 + JSON 解析修复（围栏/尾逗号/括号平衡）+ 自洽修复循环（最多 2 轮）；本地 GGUF 走 llama-server 同协议，一套代码两端 | ✅ |
| M3 | Web 端：槽位表三色来源标记 + 高影响默认标黄 + 三档并列 + 负向有效性徽章 + 自检面板；改槽位 300ms 防抖重渲染 | ✅ |
| M4 | 合成器（三档口吻 × 句式变体 × src 监督）+ 50 条基准（已入库）+ eval 四维指标 + 基座对比模式 + 导出链 merge→GGUF；`train.bat` 一键（含 ComfyUI 检查） | ✅ 管线就绪；训练执行与「两个 GGUF」需要 GPU，运行 `training/scripts/train.bat` |
| 骨架 | 骨架变体 B（换段序）/ C（换判词句）/ D（换 P3 结构）实现为 `render --skeleton`，默认骨架 A 逐字节不变；**待作者过目口吻**（§十一.2） | ✅ 待确认 |

**与计划的偏离（理由）：**

1. **M3 服务端用 stdlib `http.server`，不是 FastAPI。** 产品承诺是「双击即用」，
   零依赖是硬约束；FastAPI+uvicorn+pydantic 会把安装面扩大一个量级。
   端点形状与 FastAPI 版本一致，将来要流式输出再换不迟。
2. **Web 前端是无构建 vanilla SPA，不是 Vite+React。** 理由同上——M3 的分水岭
   是「槽位表可编辑、即时重渲染」这个交互本身，不需要引入 Node 工具链。

**M4 期间的顺带修正：** 渲染器补齐了冠词（a/an）、复数（they wear）、
时段标签表扩展（noon/dawn/evening/blue hour）——这些是场景多样性逼出来的
语法缺口；`W_MULTI_SUBJECT` 已删（渲染器现在自己处理人数，原告警已失效）。

测试：115 项，全部通过。

---

## 十二、开工前需要你确认

1. **许可证**：选 Apache-2.0（与底模、数据集都兼容）还是 GPL-3.0（与你的 OliviaComfyUi 一致）？我倾向 Apache-2.0，开源项目传染性小、更容易被集成。
2. **长档骨架变体**：除你给的那套，我再抽象 3 套（换段序 / 换判词句 / 换 P3 结构），需要你过一遍确认像不像你的写法。
3. **场景库**：30–50 个场景名词 + 每个场景的（前景道具 / 背景开口 / 材质清单 / 自然光时段），这块只有你能给，是最大的人工投入（约半天）。
4. **M0 从哪开始**：我建议先写 `packages/renderer` + `packages/negative`，因为它们决定输出质量的下限，且不依赖模型、不依赖数据、立刻能验证。