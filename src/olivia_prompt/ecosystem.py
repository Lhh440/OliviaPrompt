"""Target ecosystems and their negative-prompt behaviour.

The single most useful thing this module encodes: *most modern models do not
read negative prompts*.  Distilled / turbo / guidance-distilled checkpoints run
at cfg=1, where the negative branch is multiplied by zero.  Shipping a negative
prompt at those targets is noise, so we label the strength and, when it is
useless, convert the intent into a positive constraint patch instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple


class NegativeStrength(str, Enum):
    STRONG = "strong"   # cfg 5-7, negative really steers
    WEAK = "weak"       # some effect, keep it short
    NONE = "none"       # cfg=1 / guidance distilled: ignored
    PARAM = "param"     # not a text box; a flag (Midjourney --no)


@dataclass(frozen=True)
class Ecosystem:
    key: str
    label: str
    strength: NegativeStrength
    tag_style: str          # "booru" | "prose" | "mj"
    cfg_hint: str
    note: str

    @property
    def negative_is_text(self) -> bool:
        return self.strength in (NegativeStrength.STRONG, NegativeStrength.WEAK)


ECOSYSTEMS: Dict[str, Ecosystem] = {
    e.key: e
    for e in (
        Ecosystem(
            key="illustrious",
            label="Illustrious / NoobAI (SDXL anime)",
            strength=NegativeStrength.STRONG,
            tag_style="booru",
            cfg_hint="cfg 5-7",
            note="默认生态。负向标签影响很大，值得写满 5 层。",
        ),
        Ecosystem(
            key="sdxl",
            label="SDXL base",
            strength=NegativeStrength.STRONG,
            tag_style="booru",
            cfg_hint="cfg 5-7",
            note="负向有效，但比 Illustrious 更吃自然语言。",
        ),
        Ecosystem(
            key="sd15",
            label="SD 1.5",
            strength=NegativeStrength.STRONG,
            tag_style="booru",
            cfg_hint="cfg 7-9",
            note="负向极敏感，词表宜短。",
        ),
        Ecosystem(
            key="pony",
            label="Pony Diffusion",
            strength=NegativeStrength.STRONG,
            tag_style="booru",
            cfg_hint="cfg 6-7",
            note="自带 score_ 系列前缀，负向以 score_4/5/6 为主。",
        ),
        Ecosystem(
            key="nai",
            label="NovelAI v4",
            strength=NegativeStrength.STRONG,
            tag_style="booru",
            cfg_hint="cfg 5-6",
            note="使用 UC 预设，负向同样吃标签。",
        ),
        Ecosystem(
            key="anima",
            label="Anima",
            strength=NegativeStrength.STRONG,
            tag_style="booru",
            cfg_hint="cfg 5-7",
            note="沿用 danbooru 命名，与 Illustrious 同族。",
        ),
        Ecosystem(
            key="qwen_image",
            label="Qwen-Image",
            strength=NegativeStrength.WEAK,
            tag_style="prose",
            cfg_hint="cfg 2.5-4",
            note="吃自然语言描述，负向收益很小，短串即可。",
        ),
        Ecosystem(
            key="flux",
            label="FLUX.1 dev",
            strength=NegativeStrength.NONE,
            tag_style="prose",
            cfg_hint="guidance distilled",
            note="guidance 蒸馏，负向分支不生效，转为正向约束补丁。",
        ),
        Ecosystem(
            key="krea_turbo",
            label="Krea / SDXL Turbo / Lightning",
            strength=NegativeStrength.NONE,
            tag_style="prose",
            cfg_hint="cfg=1",
            note="cfg=1 时负向被乘零，必须留空并转正向约束。",
        ),
        Ecosystem(
            key="midjourney",
            label="Midjourney v7",
            strength=NegativeStrength.PARAM,
            tag_style="mj",
            cfg_hint="n/a",
            note="负向写成 --no a, b, c 参数，不是文本框。",
        ),
    )
}

DEFAULT_ECOSYSTEM = "illustrious"


def get_ecosystem(key: str | None) -> Ecosystem:
    key = (key or DEFAULT_ECOSYSTEM).strip().lower()
    if key in ECOSYSTEMS:
        return ECOSYSTEMS[key]
    aliases: Dict[str, str] = {
        "illustrious-xl": "illustrious",
        "illustrious_xl": "illustrious",
        "noobai": "illustrious",
        "noob": "illustrious",
        "noxl": "illustrious",
        "sdxl-base": "sdxl",
        "1.5": "sd15",
        "sd-1.5": "sd15",
        "novelai": "nai",
        "nai4": "nai",
        "flux-dev": "flux",
        "flux1": "flux",
        "krea": "krea_turbo",
        "turbo": "krea_turbo",
        "lightning": "krea_turbo",
        "mj": "midjourney",
        "midjourney-v7": "midjourney",
        "qwen": "qwen_image",
        "qwenimage": "qwen_image",
    }
    if key in aliases:
        return ECOSYSTEMS[aliases[key]]
    raise KeyError(
        f"unknown ecosystem {key!r}. Known: {', '.join(sorted(ECOSYSTEMS))}"
    )


ECOSYSTEM_CHOICES: Tuple[str, ...] = tuple(sorted(ECOSYSTEMS))


__all__ = [
    "ECOSYSTEMS",
    "ECOSYSTEM_CHOICES",
    "DEFAULT_ECOSYSTEM",
    "Ecosystem",
    "NegativeStrength",
    "get_ecosystem",
]
