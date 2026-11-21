# OliviaPrompt

**把你脑子里的画面，变成写对的正向 + 负向提示词。不需要懂英文，不需要调 API。**

本地优先的 AI 绘画提示词引擎。中文意图 → 结构化槽位 → 三档长度的正向提示词 + 五层镜像负向提示词。

> 状态：**M0 完成**。槽位 schema、三档渲染器、五层负向、自洽校验、CLI 已可用。
> 「中文需求 → 槽位」的模型在 M2/M4 接入，届时 `start.bat` 双击即用。

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

M0 只做下半部分（从槽位往下）。上半部分的模型接入是 M2，本地 GGUF 是 M4 —— 拆开做的好处是，**在还没有任何模型的时候就能验证输出质量**。

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
  render.py      三档渲染器（确定性，逐字节稳定）
  negative.py    五层镜像负向 + 生态适配
  validate.py    自洽校验规则
  ecosystem.py   目标生态与负向有效性
  cli.py         命令行
examples/        两个完整槽位示例
tests/           33 项测试
docs/            方案、schema、负向规范
```

---

## 路线图

| 阶段 | 内容 | 状态 |
|---|---|---|
| M0 | 槽位 schema + 三档渲染器 + 五层负向 + CLI | ✅ |
| M1 | 规则版批量采样（场景库 × 槽位），不跑模型 | — |
| M2 | 接入模型：中文需求 → 槽位 JSON | — |
| M3 | Web 端（槽位表可编辑、三档并列、负向有效性） | — |
| M4 | 数据合成 + Lite(4B)/Standard(8B) 训练 + eval | — |
| M5 | `start.bat` 一键分发 / docker / 模型上传 | — |

---

## 许可

Apache-2.0。词库不随仓库分发（运行时从上游按各自许可拉取），所以仓库本身是干净的。