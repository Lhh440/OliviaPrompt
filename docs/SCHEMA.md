# 槽位 Schema

`src/olivia_prompt/slots.py` 是唯一真源。渲染器、负向生成器、自洽校验全部从这里读，加一个槽位只改一处。

## 三态来源标记

不用浮点置信度 —— 模型自报的 confidence 是幻觉，不可校验、不可复现、不可训练。改用**可判定的事实**：

| `src` | 含义 | 产品行为 |
|---|---|---|
| `user` | 用户明确说了 | 直接用，绝不追问 |
| `inferred` | 模型合理推断（"赛博朋克雨夜"必然有霓虹和湿地反光） | 用，在摘要里让用户看见 |
| `default` | 无人提供，取骨架默认值 | **高影响槽则标黄提示** |

`SlotSet.open_questions()` 返回的就是「高影响 ∩ 来源为 default」的槽位 —— 也就是唯一值得问的东西。注意它关于**来源**，不是关于**空值**：一个槽位可以有很合理的默认值，但那仍然是用户没做过的决定。

## 影响权重

| 权重 | 槽位 |
|---|---|
| **high** | `subject_count` `gender` `pose_family` `setting` `action` `light_dir` `aspect` `cam_height` `medium` |
| **mid** | `age` `skin` `hair_length` `hair_color` `eye_color` `expression` `outfit_main` `perch` `handheld` `visibility` `ground` `indoor` `props` `anchor` `opening` `background` `materials` `light_time` `hit` `render` |
| **low** | `hair_style` `outfit_detail` `outfit_layer` `bartwear` `accessories` `craft` `hand_support` `hand_other` `grip` `gaze_target` `bg_far` `cool_color` `warm_color` `quality` `mood` `tags_extra` |

**追问规则**：`high ∩ src=default`，按权重排序取前 3，最多 2 轮。一题只问一个槽，必须带 2–4 个可执行选项和「你决定」兜底。

## 档位可见性

一个槽位可以只在某些档位生效：

| 只在 long | `perch` `handheld` `hand_support` `hand_other` `grip` `gaze_target` `visibility` `craft` `anchor` `opening` `bg_far` `materials` `hit` `render` |
| 只在 medium+long | `ground` `indoor` |
| 只在 short | `hair_style` `tags_extra` |
| 只在 medium | `framing` |
| 三档全用 | 其余 |

## 槽位类型

```python
Slot(
    name       = "pose_family",
    impact     = "high",
    tiers      = ("short", "medium", "long"),
    kind       = "enum",                    # str | enum | list | bool
    options    = ("sit", "stand", "walk", "lean", "kneel", "lie", "crouch"),
    default    = "sit",
    length     = 0,                         # kind=list 时的必需项数，0 = 不限
    doc        = "...",
)
```

- `kind="list"` 且 `length=3` → 少于或超过 3 项会抛 `SlotError`
- `kind="bool"` 接受 `true/false/1/0/yes/no/indoor/outdoor` 等写法
- `kind="enum"` 严格校验，`options` 空则不限

## 输入格式

两种写法可以混用，未知键直接报错（防止拼写错误被静默吞掉）：

```json
{
  "gender": "female",
  "setting": "quiet rooftop garden",
  "light_dir": { "value": "left", "src": "user" },
  "indoor": { "value": false, "src": "user" }
}
```

也可以用 `_sources` 集中声明：

```json
{
  "gender": "female",
  "_sources": { "gender": "user" }
}
```

## 姿势族决定语法

`pose_family` 是自洽校验的主轴，它同时决定动词、介词、持物落点、肩部从句、落地从句：

| 姿势 | 动词介词 | 持物落点 | 落地从句 |
|---|---|---|---|
| `sit` | sits on | resting on her lap | both feet rest naturally on the {ground} |
| `stand` | stands at | held loosely in both hands | both feet are planted evenly on the {ground} |
| `walk` | walks through | carried at her side | both feet meet the {ground} in mid-stride |
| `lean` | leans against | held against her chest | both feet rest naturally on the {ground} |
| `kneel` | kneels on | resting on her thigh | one knee rests on the {ground} |
| `lie` | lies on | resting on her chest | her body rests along the {ground} |
| `crouch` | crouches in | held close to her knees | both feet are flat on the {ground} |

代词（her/his/their）由 `gender` 驱动，包括写死在槽位文本里的第三人称（渲染时会替换）。

## 不变量

渲染器对同一份 `SlotSet` **永远输出同一串字节**。没有随机、没有变体池、没有温度。这是「用户改模板就能重渲染」的前提，也是测试能断言全文的原因。