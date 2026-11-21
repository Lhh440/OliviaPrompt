"""Negative prompt builder — five layers, mirrored from the positive prompt.

Most prompt tools ship a fixed block of ``worst quality, low quality, bad
anatomy`` boilerplate.  That is wrong for two reasons:

1. On 2026-era distilled checkpoints (turbo / lightning / guidance-distilled) the
   negative branch is multiplied by zero.  Shipping text there does nothing.
2. A long generic block *dilutes* the one or two tokens that actually suppress
   the artefact you care about.

So negatives here are **mirrored**: every clause of the positive prompt that can
fail in a specific way contributes a specific suppressant.  Holding a pencil
implies fused fingers; a single subject implies duplicate people; a "restrained
background detail" instruction implies cluttered background.

The one rule that matters most: **a negative may exclude a wrong *state*, never
another *value*.**  Writing ``silver hair`` must not put ``blonde hair`` in the
negative — that pulls the hair colour toward neutral.  ``multicolored hair``,
however, is a wrong state and belongs there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .ecosystem import Ecosystem, NegativeStrength, get_ecosystem
from .slots import SlotSet

#: Hard ceiling.  Past this the negative block stops helping and starts fighting
#: itself.  Priority order when trimming is **L1 > L3 > L4 > L2**: a style clash
#: is binary and catastrophic, whereas modern checkpoints already have decent
#: hand priors, so structural tokens are the cheapest thing to drop.
MAX_WORDS = 60

# --------------------------------------------------------------------------- #
# L1 — base layer: only what still does something on a modern checkpoint
# --------------------------------------------------------------------------- #

_L1_BASE: Tuple[str, ...] = (
    "worst quality",
    "low quality",
    "watermark",
    "text",
    "signature",
)

#: Pony-family checkpoints key off score_ tokens instead of the usual quality words.
_L1_PONY: Tuple[str, ...] = ("score_4", "score_5", "score_6")

# --------------------------------------------------------------------------- #
# L2 — structural layer: only when there is a body in frame
# --------------------------------------------------------------------------- #

_L2_BASE: Tuple[str, ...] = (
    "extra fingers",
    "fewer fingers",
    "fused fingers",
    "malformed hands",
    "poorly drawn face",
)

_L2_MULTI: Tuple[str, ...] = (
    "merged bodies",
    "conjoined figures",
    "extra heads",
    "duplicate person",
)


# --------------------------------------------------------------------------- #
# Mirror rules
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MirrorRule:
    """One positive clause -> its suppressants + its positive-constraint twin."""

    when: str
    negative: Tuple[str, ...]
    positive: Tuple[str, ...] = ()


_MIRROR: Tuple[MirrorRule, ...] = (
    MirrorRule(
        when="single_subject",
        negative=("multiple people", "duplicate subject"),
        positive=("a single subject only",),
    ),
    MirrorRule(
        when="multi_subject",
        negative=("merged bodies", "conjoined figures", "extra heads"),
        positive=(),
    ),
    MirrorRule(
        when="handheld",
        negative=("object merging into hand", "floating objects"),
        positive=("object held clearly separate from the hand",),
    ),
    MirrorRule(
        when="seated_pose",
        negative=("floating subject", "clipping through furniture"),
        positive=("subject resting in contact with the surface",),
    ),
    MirrorRule(
        when="visibility_directive",
        negative=("cropped face", "cropped hands"),
        positive=("full subject in frame", "hands fully visible"),
    ),
    MirrorRule(
        when="background_layers",
        negative=("cluttered background", "busy composition"),
        positive=("clean, uncluttered background",),
    ),
    MirrorRule(
        when="directional_light",
        negative=("flat lighting", "inconsistent shadows"),
        positive=("single directional light source", "consistent shadow direction"),
    ),
    MirrorRule(
        when="hair_color",
        # Only wrong *states*, never competing values: putting blonde/brown here
        # would drag the hair colour back toward neutral.
        negative=("multicolored hair", "hair covering eyes"),
        positive=(),
    ),
    MirrorRule(
        when="grip_quality",
        negative=(),
        positive=("correct five-fingered hands", "natural hand anatomy"),
    ),
)

_MIRROR_BY_KEY: Dict[str, MirrorRule] = {r.when: r for r in _MIRROR}


def _triggered(ss: SlotSet) -> List[str]:
    """Which mirror rules the current slot table activates."""
    keys: List[str] = []
    count = str(ss["subject_count"])
    if count == "1":
        keys.append("single_subject")
    elif count in ("2", "3"):
        keys.append("multi_subject")

    if ss.get("handheld"):
        keys.append("handheld")

    if str(ss["pose_family"]) in ("sit", "kneel", "lie", "lean", "crouch"):
        keys.append("seated_pose")

    if any(v for v in (ss.get("visibility") or ())):
        keys.append("visibility_directive")

    if ss.get("background") or ss.get("opening"):
        keys.append("background_layers")

    if ss.get("light_dir"):
        keys.append("directional_light")

    if ss.get("hair_color"):
        keys.append("hair_color")
    if ss.get("grip"):
        keys.append("grip_quality")
    return keys


# --------------------------------------------------------------------------- #
# L4 — style exclusion, mutually exclusive with the target medium
# --------------------------------------------------------------------------- #

def _style_exclusion(medium: str) -> Tuple[str, ...]:
    m = (medium or "").lower()
    if "photo" in m or "cinematic" in m:
        return ("anime", "cartoon", "illustration", "cel shading", "painterly")
    if "3d" in m or "render" in m:
        return ("2d", "flat", "sketch", "anime")
    if "painting" in m or "watercolor" in m or "oil" in m:
        return ("photograph", "3d render", "photo")
    if "pixel" in m:
        return ("antialiasing", "smooth shading")
    # anime / illustration / default
    return ("photorealistic", "3d render", "photo")


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #

@dataclass
class NegativeResult:
    text: str
    strength: str
    layers: Dict[str, List[str]] = field(default_factory=dict)
    positive_patch: List[str] = field(default_factory=list)
    mj_param: str = ""
    note: str = ""
    trimmed: int = 0

    @property
    def is_text(self) -> bool:
        return bool(self.text)


def _fit(blocks: Sequence[Sequence[str]], max_words: int = MAX_WORDS) -> Tuple[List[str], int]:
    """Assemble in priority order, stopping before the word ceiling."""
    out: List[str] = []
    seen = set()
    dropped = 0
    words = 0
    for block in blocks:
        for tag in block:
            tag = tag.strip()
            if not tag or tag in seen:
                continue
            cost = len(tag.split())
            if words + cost > max_words:
                dropped += 1
                continue
            seen.add(tag)
            out.append(tag)
            words += cost
    return out, dropped


def build_negative(
    slots: SlotSet,
    ecosystem: str | Ecosystem | None = None,
    max_words: int = MAX_WORDS,
) -> NegativeResult:
    """Build the 5-layer negative prompt for the target ecosystem."""
    eco = ecosystem if isinstance(ecosystem, Ecosystem) else get_ecosystem(ecosystem)
    keys = _triggered(slots)

    l1: List[str] = list(_L1_BASE)
    if eco.key == "pony":
        l1 = list(_L1_PONY) + ["worst quality", "low quality", "jpeg artifacts", "watermark"]

    l2: List[str] = list(_L2_BASE)
    if "multi_subject" in keys:
        l2.extend(_L2_MULTI)

    l3: List[str] = []
    patch: List[str] = []
    for key in keys:
        rule = _MIRROR_BY_KEY.get(key)
        if rule is None:
            continue
        l3.extend(rule.negative)
        patch.extend(rule.positive)

    l4 = list(_style_exclusion(str(slots["medium"])))

    layers = {"L1": l1, "L2": l2, "L3": l3, "L4": l4}
    strength = eco.strength.value

    # --- ecosystems where a text negative does nothing --------------------
    if eco.strength is NegativeStrength.NONE:
        return NegativeResult(
            text="",
            strength=strength,
            layers=layers,
            positive_patch=_dedupe(patch),
            note=(
                f"{eco.label} 在 {eco.cfg_hint} 下负向分支被乘零，文本负向无效。"
                "已把抑制意图转成正向约束补丁，请把它追加到正向提示词末尾。"
            ),
        )

    # --- Midjourney: a flag, not a text box ------------------------------
    if eco.strength is NegativeStrength.PARAM:
        chosen, trimmed = _fit((l1, l3, l4, l2), max_words)
        param = f"--no {', '.join(chosen)}" if chosen else ""
        return NegativeResult(
            text=param,
            strength=strength,
            layers=layers,
            positive_patch=[],
            mj_param=param,
            note="Midjourney 不支持负向文本框，已转成 --no 参数，追加到正向末尾。",
            trimmed=trimmed,
        )

    # --- normal text negative --------------------------------------------
    # Weak targets keep it shorter: the signal is already marginal.
    budget = max_words if eco.strength is NegativeStrength.STRONG else min(max_words, 18)
    chosen, trimmed = _fit((l1, l3, l4, l2), budget)
    note = ""
    if eco.strength is NegativeStrength.WEAK:
        note = f"{eco.label} 负向收益有限（{eco.cfg_hint}），已压缩到 {budget} 词以内。"
    elif trimmed:
        note = f"已按优先级截断 {trimmed} 项，控制在 {max_words} 词内。"
    return NegativeResult(
        text=", ".join(chosen),
        strength=strength,
        layers=layers,
        positive_patch=[],
        note=note,
        trimmed=trimmed,
    )


def _dedupe(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for i in items:
        i = i.strip()
        if i and i not in seen:
            seen.add(i)
            out.append(i)
    return out


__all__ = ["build_negative", "NegativeResult", "MAX_WORDS"]
