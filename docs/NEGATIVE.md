# 负向提示词规范

大多数人写的负向是一串固定词表：`worst quality, low quality, bad anatomy, extra fingers, ...`。这在两个层面是错的。

1. **在 2026 年的蒸馏模型上基本无效**。turbo / lightning / guidance-distilled 跑 cfg=1，负向分支被乘零。给它文本负向是纯噪音。
2. **词表越长越有害**。负向的作用是压制**特定**伪影，通用词堆得越多，真正该压的那两三个 token 被稀释得越厉害。

所以这里的负向是**镜像**出来的：每一层都从正向的具体内容反推。

---

## 五层结构

| 层 | 内容 | 什么时候加 |
|---|---|---|
| **L1** 基底 | `worst quality, low quality, watermark, text, signature` | 始终，且只有 5 个 |
| **L2** 结构 | `extra fingers, fewer fingers, fused fingers, malformed hands, poorly drawn face` | 画面里有身体时 |
| **L3** 镜像 | 从正向逐句反推 | 见下表 |
| **L4** 风格 | 与目标媒介互斥的那一类 | 始终 |
| **L5** 长度纪律 | 硬截断 60 词 | 始终 |

**裁剪优先级：L1 > L3 > L4 > L2。** 风格冲突是二元的、灾难性的；而现代 checkpoint 的手部先验已经不错，结构词是最便宜的牺牲品。

---

## L3 镜像规则

| 正向写了什么 | 负向加什么 |
|---|---|
| `a single adult woman` | `multiple people, duplicate subject` |
| 人数 2–3 | `merged bodies, conjoined figures, extra heads` |
| `holding a pencil` / 任何 handheld | `object merging into hand, floating objects` |
| `sits on` / 坐跪躺倚蹲 | `floating subject, clipping through furniture` |
| `Keep her face, hands and X clearly visible` | `cropped face, cropped hands` |
| `opens onto layers of distant rooftops` | `cluttered background, busy composition` |
| `sunlight enters from the left` | `flat lighting, inconsistent shadows` |
| `long silver hair` | `multicolored hair, hair covering eyes` |
| `relaxed, believable grip` | （无负向，只贡献正向补丁） |

---

## 唯一的判断标准

> **负向只能排除「错误状态」，不能排除「其他取值」。**

这是区分负向写得好不好的分水岭：

```diff
  正向：long silver hair
- 负向：blonde hair, brown hair, black hair      ❌ 会把发色拉回中性
+ 负向：multicolored hair, gradient hair, hair covering eyes   ✅ 只排错误状态
```

同理，`solo` 的正确负向是 `multiple people`（错误状态），而不是把每个可能的角色名塞进去。

代码里这条约束写在 `negative.py` 的注释和测试 `test_never_excludes_other_values` 里，改坏了会直接挂测试。

---

## 生态适配

负向的有效性完全取决于目标 checkpoint 的 cfg：

| 生态 | 强度 | 输出形态 |
|---|---|---|
| `illustrious` / `sdxl` / `sd15` / `nai` / `anima` | strong | 完整五层串 |
| `pony` | strong | L1 换成 `score_4, score_5, score_6` |
| `qwen_image` | weak | 压缩到 18 词以内 |
| `flux` / `krea_turbo` | **none** | **留空** + 正向约束补丁 |
| `midjourney` | param | `--no a, b, c` |

### 负向无效时怎么办

不交付死文本，而是把抑制意图翻译回正向：

```
L3 想压制的                          → 补丁里写的
multiple people / duplicate subject  → a single subject only
object merging into hand             → object held clearly separate from the hand
cropped face / cropped hands         → full subject in frame, hands fully visible
cluttered background                 → clean, uncluttered background
flat lighting / inconsistent shadows → single directional light source, consistent shadow direction
malformed hands                      → correct five-fingered hands, natural hand anatomy
floating subject                     → subject resting in contact with the surface
```

这样"正反向成套"对每个生态都有意义，而不是对一半用户交付无用文本。

---

## 不该出现的词

已被现代模型免疫、纯占预算的：

```
normal quality, lowres, bad proportions, ugly, duplicate,
bad hands (已被 five fingers 系取代),
ugly face, deformed, disfigured, worst quality（只保留 1 个）
```

`jpeg artifacts` 和 `blurry` 已从 L1 移除 —— 现代采样器极少产生这两种伪影，占位不划算。