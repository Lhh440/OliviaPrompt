"""Slot registry — the single source of truth for what OliviaPrompt decides.

A *slot* is one decision the model (or the user) has to make before a prompt can
be written.  Slots are declared once here and consumed by every renderer, the
negative builder and the self-consistency validator, so that adding a slot never
requires touching three files.

Design notes
------------
* No floating-point confidence.  LLMs report confidence as a hallucination;
  :class:`SlotSet` instead tracks a *source* that is a checkable fact:
  ``user`` / ``inferred`` / ``default``.
* Every slot has a default, so a partial input always renders.
* Impact is the axis the product uses to decide what is worth asking about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple

TIERS: Tuple[str, ...] = ("short", "medium", "long")

IMPACT_HIGH = "high"
IMPACT_MID = "mid"
IMPACT_LOW = "low"
IMPACT_ORDER = {IMPACT_HIGH: 0, IMPACT_MID: 1, IMPACT_LOW: 2}

#: Where a slot's value came from.  Deliberately a closed enum of facts.
SRC_USER = "user"
SRC_INFERRED = "inferred"
SRC_DEFAULT = "default"
SRC_LABELS = {SRC_USER: "用户指定", SRC_INFERRED: "模型推断", SRC_DEFAULT: "骨架默认"}


@dataclass(frozen=True)
class Slot:
    """One decision unit."""

    name: str
    impact: str
    tiers: Tuple[str, ...]
    kind: str = "str"          # "str" | "enum" | "list" | "bool"
    options: Tuple[str, ...] = ()
    default: Any = None
    length: int = 0            # for kind == "list": required item count
    doc: str = ""

    def in_tier(self, tier: str) -> bool:
        return tier in self.tiers


def _s(
    name: str,
    impact: str,
    tiers: Tuple[str, ...],
    kind: str = "str",
    options: Tuple[str, ...] = (),
    default: Any = None,
    length: int = 0,
    doc: str = "",
) -> Slot:
    return Slot(
        name=name,
        impact=impact,
        tiers=tuple(tiers),
        kind=kind,
        options=tuple(options),
        default=default,
        length=length,
        doc=doc,
    )


_ALL: Tuple[str, ...] = TIERS

# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_SLOT_LIST: Tuple[Slot, ...] = (
    # ---- identity ----------------------------------------------------------
    _s("subject_count", IMPACT_HIGH, _ALL, kind="enum", options=("1", "2", "3"), default=1,
       doc="主体人数。短档渲染成 1girl；长档决定 a single / two。"),
    _s("gender", IMPACT_HIGH, _ALL, kind="enum", options=("female", "male", "other"),
       default="female", doc="决定代词与 danbooru 主体标签。"),
    _s("age", IMPACT_MID, _ALL, default="adult",
       doc="年龄形容词。默认 adult。"),
    _s("skin", IMPACT_MID, _ALL, default="fair-skinned",
       doc="肤色形容词，长档 P1a 使用。"),
    _s("hair_length", IMPACT_MID, _ALL, kind="enum",
       options=("long", "medium", "short", "very long"), default="long"),
    _s("hair_color", IMPACT_MID, _ALL, default="silver",
       doc="同时驱动 P4a 的受光面描述。"),
    _s("hair_style", IMPACT_LOW, ("short",), default="straight"),
    _s("eye_color", IMPACT_MID, _ALL, default="blue-gray"),
    _s("expression", IMPACT_MID, _ALL, default="calm, attentive"),

    # ---- wardrobe ----------------------------------------------------------
    _s("outfit_main", IMPACT_MID, _ALL, default="deep navy dress",
       doc="主衣：颜色 + 材质 + 单品，例：deep navy dress。"),
    _s("outfit_detail", IMPACT_LOW, _ALL, default="delicate ivory embroidery",
       doc="主衣上的工艺/纹样。"),
    _s("outfit_layer", IMPACT_LOW, _ALL, default="lightweight cream cardigan",
       doc="叠穿层。"),
    _s("bartwear", IMPACT_LOW, _ALL, default="simple leather ankle boots",
       doc="鞋履。"),
    _s("accessories", IMPACT_LOW, _ALL, kind="list",
       doc="配饰列表，可为空。"),
    _s("craft", IMPACT_LOW, ("long",), kind="list", length=2,
       default=("stitched hems", "softly folded fabric"),
       doc="P1c 判词句里的两个微观工艺细节。"),

    # ---- pose / action -----------------------------------------------------
    _s("pose_family", IMPACT_HIGH, _ALL, kind="enum",
       options=("sit", "stand", "walk", "lean", "kneel", "lie", "crouch"),
       default="sit",
       doc="姿势族。决定动词、介词、地面接触从句，是自洽校验的主轴。"),
    _s("perch", IMPACT_MID, ("long",), default="wooden bench",
       doc="所坐 / 所倚的承载体。"),
    _s("setting", IMPACT_HIGH, _ALL, default="quiet rooftop garden",
       doc="场景。高影响槽：定错了整张图废掉。"),
    _s("action", IMPACT_HIGH, _ALL, default="drawing in an open sketchbook",
       doc="主体在做的事，作为 P2a 的现在分词短语。"),
    _s("handheld", IMPACT_MID, ("long",), default="open sketchbook",
       doc="手中持有物。出现在镜像负向里（防穿透 / 融合）。"),
    _s("hand_support", IMPACT_LOW, ("long",), default="pages",
       doc="支撑手接触的部位。"),
    _s("hand_other", IMPACT_LOW, ("long",), default="holds a pencil",
       doc="另一只手在做什么。"),
    _s("grip", IMPACT_LOW, ("long",), default="relaxed, believable grip",
       doc="握持质感判词，是抗手部伪影的关键。"),
    _s("gaze_target", IMPACT_LOW, ("long",), default="the drawing",
       doc="视线落点。"),
    _s("visibility", IMPACT_MID, ("long",), kind="list", length=3,
       default=("face", "hands", "sketchbook"),
       doc="P2c 显式可见性指令的三项。"),
    _s("ground", IMPACT_MID, ("medium", "long"), default="stone paving",
       doc="地面材质。坐 / 跪 / 躺姿势必需。"),

    # ---- world -------------------------------------------------------------
    _s("indoor", IMPACT_MID, ("medium", "long"), kind="bool", default=False,
       doc="室内则为 True。用于自洽校验：室内不应出现树叶细脉这类户外材质。"),
    _s("props", IMPACT_MID, _ALL, kind="list", length=3,
       default=("Climbing roses", "terracotta planters", "slender grasses"),
       doc="P3a 的三个前景框景道具。"),
    _s("anchor", IMPACT_MID, ("long",), default="bench",
       doc="P3a 被框住的那个物体（通常是 perch 的简称）。"),
    _s("opening", IMPACT_MID, ("long",), default="low wrought-iron railing",
       doc="P3b 的开口物（栏杆 / 拱窗 / 门洞），负责做纵深。"),
    _s("background", IMPACT_MID, _ALL, default="distant rooftops",
       doc="背景主体。"),
    _s("bg_far", IMPACT_LOW, ("long",), default="city",
       doc="P3c 细节衰减的终点名词。"),
    _s("materials", IMPACT_MID, ("long",), kind="list", length=3,
       default=("wood", "nearby leaves", "stone"),
       doc="P3c 的三种材质，对应木纹 / 叶脉 / 石面。"),

    # ---- light -------------------------------------------------------------
    _s("light_time", IMPACT_MID, _ALL, default="Warm late-afternoon",
       doc="光照时段。用于短档的光照标签推断。"),
    _s("light_dir", IMPACT_HIGH, _ALL, kind="enum",
       options=("left", "right", "front", "top", "back-right", "back-left"),
       default="left",
       doc="光向。缺它会导致光影自相矛盾，是高影响槽。"),
    _s("hit", IMPACT_MID, ("long",), kind="list", length=2,
       default=("edges of her silver hair", "embroidery of her dress"),
       doc="P4a 光打在什么上，两项。"),
    _s("cool_color", IMPACT_LOW, ("long",), default="blue",
       doc="阴影侧冷色。"),
    _s("warm_color", IMPACT_LOW, ("long",), default="amber",
       doc="受光侧暖色。"),

    # ---- camera ------------------------------------------------------------
    _s("aspect", IMPACT_HIGH, _ALL, default="spacious horizontal",
       doc="画幅。也用于自洽校验：画幅与姿势族需相容。"),
    _s("cam_height", IMPACT_HIGH, _ALL, default="seated eye level",
       doc="机位高度。与姿势族冲突会被自洽校验拦下。"),
    _s("framing", IMPACT_MID, ("medium",), default="mid-shot",
       doc="景别。仅短档 / 中档使用。"),

    # ---- style -------------------------------------------------------------
    _s("medium", IMPACT_HIGH, _ALL, default="anime illustration",
       doc="媒介：anime illustration / cinematic photography / oil painting / 3d render。"),
    _s("model_style", IMPACT_MID, _ALL, default="krea2 style",
       doc="模型风格触发词。为空则前缀行只出 quality + medium。"),
    _s("quality", IMPACT_LOW, _ALL, kind="list", length=2,
       default=("masterpiece", "very aesthetic"),
       doc="前缀行质量词。由渲染器注入，不让模型生成。"),
    _s("render", IMPACT_MID, ("long",), kind="list", length=3,
       default=("Refined linework", "nuanced skin shading", "distinct material textures"),
       doc="P4d 的三个渲染描述。"),
    _s("mood", IMPACT_LOW, _ALL, default="peaceful, coherent",
       doc="氛围判词，出现在 P4d 与中档末尾。"),
    _s("tags_extra", IMPACT_LOW, ("short",), kind="list",
       doc="短档额外追加的 danbooru 标签。"),
)

SLOTS: Dict[str, Slot] = {s.name: s for s in _SLOT_LIST}
SLOT_ORDER: Tuple[str, ...] = tuple(s.name for s in _SLOT_LIST)


# --------------------------------------------------------------------------- #
# SlotSet
# --------------------------------------------------------------------------- #

class SlotError(ValueError):
    """Raised when a slot payload cannot be interpreted."""


class SlotSet:
    """A resolved slot table: values + provenance, with defaults filled in."""

    __slots__ = ("_values", "_sources", "tier")

    def __init__(self, values: Mapping[str, Any], sources: Mapping[str, str] | None = None,
                 tier: str = "long"):
        self.tier = tier
        self._values: Dict[str, Any] = {}
        self._sources: Dict[str, str] = {}
        sources = dict(sources or {})
        for name, slot in SLOTS.items():
            if name in values and values[name] is not None:
                self._values[name] = _coerce(slot, values[name])
                src = sources.get(name, SRC_INFERRED)
            else:
                self._values[name] = slot.default
                src = SRC_DEFAULT
            if src not in SRC_LABELS:
                raise SlotError(f"slot {name!r}: unknown source {src!r}")
            self._sources[name] = src

    # -- construction -------------------------------------------------------
    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], tier: str = "long") -> "SlotSet":
        """Build from a loose dict.

        Accepts either shape, mixed freely::

            {"gender": "female", "setting": "..."}                  # values only
            {"gender": {"value": "female", "src": "user"}, ...}     # explicit
        """
        if not isinstance(raw, Mapping):
            raise SlotError("slot payload must be a mapping")
        values: Dict[str, Any] = {}
        sources: Dict[str, str] = {}
        declared = raw.get("_sources") if isinstance(raw.get("_sources"), Mapping) else {}
        for key, val in raw.items():
            if key == "_sources" or not isinstance(key, str) or key.startswith("_"):
                continue
            if key not in SLOTS:
                raise SlotError(
                    f"unknown slot {key!r}. Known slots: {', '.join(SLOT_ORDER)}"
                )
            if isinstance(val, Mapping) and "value" in val:
                values[key] = val.get("value")
                sources[key] = str(val.get("src", SRC_INFERRED))
            else:
                values[key] = val
                if key in declared:
                    sources[key] = str(declared[key])
        return cls(values, sources, tier=tier)

    @classmethod
    def defaults(cls, tier: str = "long") -> "SlotSet":
        return cls({}, {}, tier=tier)

    # -- access -------------------------------------------------------------
    def __getitem__(self, name: str) -> Any:
        return self._values[name]

    def get(self, name: str, fallback: Any = None) -> Any:
        return self._values.get(name, fallback)

    def source(self, name: str) -> str:
        return self._sources.get(name, SRC_DEFAULT)

    def is_default(self, name: str) -> bool:
        return self.source(name) == SRC_DEFAULT

    def values(self) -> Dict[str, Any]:
        return dict(self._values)

    def sources(self) -> Dict[str, str]:
        return dict(self._sources)

    # -- product helpers ----------------------------------------------------
    def open_questions(self) -> List[Slot]:
        """High-impact slots nobody actually specified.

        Note this is about *provenance*, not emptiness: a slot may carry a
        sensible skeleton default and still be a decision the user never made.
        Those are exactly the questions worth asking.
        """
        return [
            s for s in _SLOT_LIST
            if s.impact == IMPACT_HIGH
            and s.in_tier(self.tier)
            and self.is_default(s.name)
        ]

    def pronoun(self, kind: str = "subj") -> str:
        plural = str(self._values.get("subject_count")) not in ("1", "0", None)
        if plural:
            return {"subj": "they", "obj": "them", "poss": "their"}[kind]
        g = self._values.get("gender")
        if g == "male":
            return {"subj": "he", "obj": "him", "poss": "his"}[kind]
        if g == "other":
            return {"subj": "they", "obj": "them", "poss": "their"}[kind]
        return {"subj": "she", "obj": "her", "poss": "her"}[kind]

    def as_dict(self) -> Dict[str, Any]:
        """Round-trippable payload using the explicit ``{"value","src"}`` shape."""
        return {
            name: {
                "value": _plain(self._values[name]),
                "src": self._sources[name],
                "impact": SLOTS[name].impact,
            }
            for name in SLOT_ORDER
            if SLOTS[name].in_tier(self.tier)
        }

    def __repr__(self) -> str:  # pragma: no cover
        filled = sum(1 for v in self._values.values() if v not in (None, "", (), []))
        return f"<SlotSet tier={self.tier} filled={filled}/{len(SLOTS)}>"


def _plain(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


def _coerce(slot: Slot, value: Any) -> Any:
    if slot.kind == "list":
        if isinstance(value, str):
            items = [p.strip() for p in value.split(",") if p.strip()]
        elif isinstance(value, Iterable):
            items = [str(v).strip() for v in value if str(v).strip()]
        else:
            raise SlotError(
                f"slot {slot.name!r} expects a list, got {type(value).__name__}"
            )
        if slot.length and len(items) != slot.length:
            raise SlotError(
                f"slot {slot.name!r} expects exactly {slot.length} items, got {len(items)}"
            )
        return tuple(items)
    if slot.kind == "bool":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("1", "true", "yes", "y", "on", "indoor", "室内"):
            return True
        if text in ("0", "false", "no", "n", "off", "outdoor", "室外"):
            return False
        raise SlotError(f"slot {slot.name!r} expects a boolean, got {value!r}")
    if slot.kind == "enum":
        text = str(value).strip()
        if slot.options and text not in slot.options:
            raise SlotError(
                f"slot {slot.name!r} must be one of {slot.options}, got {text!r}"
            )
        return text
    return str(value).strip()


__all__ = [
    "SLOTS",
    "SLOT_ORDER",
    "TIERS",
    "Slot",
    "SlotSet",
    "SlotError",
    "IMPACT_HIGH",
    "IMPACT_MID",
    "IMPACT_LOW",
    "IMPACT_ORDER",
    "SRC_USER",
    "SRC_INFERRED",
    "SRC_DEFAULT",
    "SRC_LABELS",
]
