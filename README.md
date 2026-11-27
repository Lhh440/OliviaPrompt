# OliviaPrompt

**把你脑子里的画面，变成写对的正向 + 负向提示词。不需要懂英文，不需要调 API。**

本地优先的 AI 绘画提示词引擎。中文意图 → 结构化槽位 → 三档长度的正向提示词 + 五层镜像负向提示词。

> 状态：**M0–M5 完成**。槽位 schema、三档渲染器（含 4 种长档骨架）、五层负向、自洽校验、
> 规则版批量采样、模型接入（OpenAI 兼容）、Web 端、数据合成与训练管线全部可用。
> **Lite 模型已训练并上架 Hugging Face**：[Lhh1sillho/OliviaPrompt-GGUF](https://huggingface.co/Lhh1sillho/OliviaPrompt-GGUF)。

---

## 为什么需要它

GitHub 上现有的提示词工具分成两类，都不解决真正的问题：

- **标签超市**（`danbooru-diffusion-prompt-builder` 等）——手动点选标签，不解需求
- **提示词壳**（`prompt-forge` 等）——一句话生成提示词，但**要用户自己接一个 OpenAI 兼容接口**

OliviaPrompt 的区别是把**模型一起发出去**：下载、双击、浏览器打开就能用。用户不需要有 API、不需要懂「什么模型适合写提示词」。

---

## 三个不一样的地方

**1. 模型不写最终字符串，只填槽位。**

格式由代码保证，不由模型保证。所以三档长度几乎零成本（同一份槽位，三个渲染模板），格式永远不会崩，而且**用户能改模板 fork 自己的写法，不用重训模型**。

**2. 负向是镜像出来的，不是背词表。**

每一层负向都从正向的具体内容反推：正向写了 `holding a pencil`，负向才有 `object merging into hand`。而且 —— **负向只排除错误状态，绝不排除其他取值**。写了 `silver hair` 不会往负向塞 `blonde hair`（那会把发色拉回中性）。

**3. 负向有效性按生态判定。**

cfg=1 的蒸馏模型（Krea Turbo / Lightning / Flux）负向分支被乘零，给它们文本负向等于噪音。这种情况下引擎**留空，并自动把抑制意图转成正向约束补丁**：

```
正向: ... a single subject only, correct five-fingered hands, natural hand anatomy
```

---

## 快速开始

```bash
git clone https://github.com/Lhh440/OliviaPrompt.git
cd OliviaPrompt
pip install -e .

# 从示例槽位渲染长档
olivia-prompt render examples/slots_rooftop.json --tier long

# 三档一次出全
olivia-prompt render examples/slots_rooftop.json --all

# 生成一份可编辑的槽位模板
olivia-prompt init --tier medium > my_slots.json

# 看所有槽位定义
olivia-prompt slots

# 看目标生态与负向有效性
olivia-prompt eco
```

不开 venv 也行，直接跑模块：

```bash
set PYTHONPATH=src
python -m olivia_prompt.cli render examples/slots_rooftop.json --tier long
```

### 中文需求 → 提示词（M2，接任意 OpenAI 兼容端点）

```bash
# 环境变量或 ~/.olivia-prompt/config.json 配置端点与模型
set OLIVIA_BASE_URL=http://127.0.0.1:8081/v1
set OLIVIA_MODEL=olivia-lite

olivia-prompt ask "赛博朋克雨夜巷子抽烟的银发女" --tier long --show-slots
# 本地 GGUF：llama-server -m models/olivia-lite-q4_k_m.gguf --port 8081 即可
```

### Web 端（M3，槽位表可编辑 + 三档并列 + 负向有效性标注）

```bash
olivia-prompt web --open        # 零依赖 stdlib 服务器，http://127.0.0.1:7860
```

左栏输入中文需求生成槽位 → 中栏槽位表（user/inferred/default 三色来源标记，
高影响+默认 标黄）→ 右栏三档并列输出 + 负向有效性徽章 + 自检。
**改一个槽位只重渲染，不重跑模型。**

### 规则版批量采样（M1，不跑模型）

```bash
olivia-prompt scenes                          # 60 个结构化场景（场景库 × 材质 × 光时段）
olivia-prompt sample --n 200 --render --out samples   # 批量出 200 条三档正反向
olivia-prompt coverage --n 200                # 槽位覆盖矩阵（预训练门禁）
```

### 训练（M4，需 GPU）

```bash
training\scripts\train.bat lite       # Qwen3-4B：合成数据 → LLaMA-Factory → GGUF Q4_K_M
training\scripts\train.bat standard   # Qwen3-8B：Q5_K_M，长档质量更高
python training/scripts/eval.py --base-url http://127.0.0.1:8081/v1 --model olivia --label lite
```

Windows 实测版本组合与踩坑记录见 [docs/DATA.md](docs/DATA.md)。

### 一键使用（M5）

```bash
scripts\start.bat            # 双击即用：装包 → 按 models/models.json 下载 GGUF
                             # （断点续传 + sha256 校验）→ 下载 llama-server →
                             # 起 8081 推理 + 7860 Web → 打开浏览器
scripts\start.bat standard   # 用 8B 档
```

大文件默认落在 `.\models\`（可用环境变量 `OLIVIA_MODELS_DIR` 指到其他盘）。
Docker 方式：`docker compose up`（llama.cpp server-cuda 推理 + Web 两个服务）。
权重文件不入仓库；上传用 `scripts/upload_models.py`（需 HF_WRITE_TOKEN）。

---

## 训练成果（Lite 4B，50 条基准实测）

微调模型 vs 原始基座（few-shot）在同一份 50 case 基准上的表现：

```
metric                      lite         base
json_ok_rate              100.0%        60.0%     输出永远是合法槽位 JSON
slot_acc                   98.0%        52.2%     44 个槽位的值准确率
weighted_acc               97.8%        55.2%     高影响槽位加权
consistency_error_rate      0.0%         6.7%     渲染前自洽校验错误率
```

「模型只填槽位、格式由代码保证」的产品假设在 4B 上成立：模型从未输出过
非法 JSON，人类可读的三档提示词全部由确定性渲染器生成。

### 模型下载

| 文件 | 大小 | 说明 |
|---|---|---|
| [olivia-lite-qwen3-4b-q4_k_m.gguf](https://huggingface.co/Lhh1sillho/OliviaPrompt-GGUF/resolve/main/olivia-lite-qwen3-4b-q4_k_m.gguf) | 2.5GB | **成品**，短+中档为主，长档可用 |
| [qwen3-4b-base-q4_k_m.gguf](https://huggingface.co/Lhh1sillho/OliviaPrompt-GGUF/resolve/main/qwen3-4b-base-q4_k_m.gguf) | 2.5GB | 原始 Qwen3-4B（基座对照） |

sha256 见 [models/models.json](models/models.json)，`scripts/start.bat` 下载时自动校验。
国内直连慢可用 `set HF_ENDPOINT=https://hf-mirror.com` 再跑 start.bat。

---

## 三档输出

同一份槽位，三种形态。

**short** —— booru 标签串，给 Illustrious / NoobAI / Pony / NAI

```
masterpiece, very aesthetic, illustration, 1girl, solo, long_hair, silver_hair,
blue_eyes, calm, navy_dress, cardigan, sitting, eye_level, wide_shot, rooftop,
garden, afternoon, from_left
```

**medium** —— 三段散文，~150 词

```
An anime illustration of a single fair-skinned adult woman with long silver hair,
blue-gray eyes and a calm, attentive expression, wearing a deep navy dress with
delicate ivory embroidery, a lightweight cream cardigan and simple leather ankle boots.

She sits on a wooden bench in a quiet rooftop garden, drawing in an open sketchbook.
Her gaze follows the drawing, her shoulders are slightly inclined forward, and both
feet rest naturally on the stone paving.

Warm late-afternoon sunlight enters from the left, catching edges of her silver hair.
Climbing roses and terracotta planters frame the bench, opening onto distant rooftops.
Use a spacious horizontal composition at seated eye level; refined linework and
nuanced skin shading create a peaceful, coherent illustration.
```

**long** —— 四段 / 十四个句式位，~250 词

前缀行 → 主体与服饰（含质感判词）→ 动作与物理可信性（含可见性指令）→ 环境与材质（含细节衰减）→ 光、色、构图、渲染判词。

```bash
olivia-prompt render examples/slots_rooftop.json --tier long
```

---

## 目标生态

```bash
olivia-prompt eco
```

| key | 负向强度 | 说明 |
|---|---|---|
| `illustrious` ★ | strong | 默认。Illustrious / NoobAI 系 |
| `sdxl` / `sd15` | strong | 负向有效 |
| `pony` | strong | 自动改用 `score_4/5/6` |
| `nai` / `anima` | strong | 同 danbooru 命名 |
| `qwen_image` | weak | 压缩到 18 词以内 |
| `flux` / `krea_turbo` | none | 留空，转正向约束补丁 |
| `midjourney` | param | 转 `--no a, b, c` |

---

## 工作流

```
中文需求
   │
   ├─[模型]─▶ 槽位 JSON（含 user / inferred / default 来源标记）
   │            │
   │            ├─[代码] 自洽校验：坐姿↔落地从句、室内↔树叶细脉、光向↔阴影
   │            ├─[代码] 渲染器 ×3  ──▶ short / medium / long
   │            ├─[代码] 负向 5 层  ──▶ 反向提示词
   │            └─[代码] 生态适配    ──▶ 有效性标注 / --no / 正向补丁
   │
   └─▶ Web：槽位表可编辑 ──▶ 改一个槽 ──▶ 只重渲染，不重跑模型
```

上半部分（模型接入）与下半部分（渲染/负向/适配）拆开做的好处是，**在还没有任何模型的时候就能验证输出质量**——M0 的 33 项测试就是那时候写下的；如今 M2 已把上半部分接上，模型只填槽位，其他一切由代码保证。

---

## 自洽校验

槽位由模型独立决策，必然自相矛盾。这些规则替模型把关：

```bash
olivia-prompt check examples/slots_rooftop.json
```

```
[ERROR] E_CAM_POSE (cam_height): cam_height='seated eye level' 与姿势族 stand 冲突
[WARN ] W_INDOOR_MATERIAL (materials): 室内场景出现了户外材质 ['nearby leaves']
[WARN ] W_HANDHELD_ACTION (action): handheld='open sketchbook' 未在 action 中提及
```

---

## 仓库结构

```
src/olivia_prompt/
  slots.py       槽位注册表（唯一真源）
  render.py      三档渲染器（确定性，逐字节稳定；长档支持骨架变体 A-D）
  skeletons.py   长档骨架变体 B/C/D（换段序 / 换判词句 / 换 P3 结构）
  negative.py    五层镜像负向 + 生态适配
  validate.py    自洽校验规则
  ecosystem.py   目标生态与负向有效性
  scenes.py      场景库加载器（data/scenes.json，60 场景）
  pools.py       人物/服饰/风格池（含中文对照，供数据合成）
  sample.py      M1 规则版采样器 + 覆盖矩阵
  llm/           M2 接入层（OpenAI 兼容客户端 / 槽位填充 / 修复循环）
  webapp/        M3 Web 端（stdlib 服务器 + 无构建 SPA）
  cli.py         命令行
data/            场景库 + 活动池（结构化的「场景库.txt」）
training/        M4：数据合成器 / LLaMA-Factory 配置 / 导出 / 评测 / 50 条基准
examples/        两个完整槽位示例
tests/           115 项测试
docs/            方案、schema、负向规范、数据与训练
```

---

## 路线图

| 阶段 | 内容 | 状态 |
|---|---|---|
| M0 | 槽位 schema + 三档渲染器 + 五层负向 + CLI | ✅ |
| M1 | 规则版批量采样（场景库 × 槽位），不跑模型 | ✅ 60 场景 / 覆盖矩阵 / 200 条验收 |
| M2 | 接入模型：中文需求 → 槽位 JSON（OpenAI 兼容 + 本地 GGUF 同协议） | ✅ 含自洽修复循环 |
| M3 | Web 端（槽位表可编辑、三档并列、负向有效性） | ✅ 零依赖 stdlib 服务器 |
| M4 | 数据合成 + Lite(4B) 训练 + eval | ✅ 1083 步，槽位准确率 98%（基座 52%）；Standard(8B) 配置就绪，待 ≥24GB 显存（[docs/DATA.md](docs/DATA.md)） |
| M5 | `start.bat` 一键分发 / docker / 模型上传 | ✅ 模型已上架 HF |

---

## 许可

Apache-2.0。词库不随仓库分发（运行时从上游按各自许可拉取），所以仓库本身是干净的。