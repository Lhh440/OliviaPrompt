"""Intent pipeline: 中文需求 → 槽位 → 三档正向 + 五层负向 + 自检.

This is the one call the web UI makes.  The model runs once (plus repair
rounds); every downstream artifact is deterministic code, so "改一个槽只重渲染"
stays true even when the slots came from a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..ecosystem import Ecosystem, get_ecosystem
from ..negative import build_negative
from ..render import render_all
from ..slots import SlotSet
from ..validate import Issue, validate
from .client import ChatClient, ChatConfig, resolve_config
from .slotfill import FillResult, fill_slots


@dataclass
class PipelineResult:
    intent: str
    tier: str
    slots: SlotSet
    issues: List[Issue]
    prompts: Dict[str, Dict[str, object]]
    fill: Optional[FillResult] = None
    notes: List[str] = field(default_factory=list)

    @property
    def rounds(self) -> int:
        return self.fill.rounds if self.fill else 0


def run_intent(
    intent: str,
    tier: str = "long",
    ecosystem: str | Ecosystem | None = None,
    client: Optional[ChatClient] = None,
    config: Optional[ChatConfig] = None,
    max_rounds: int = 2,
    skeleton: str = "A",
    compact: bool = False,
) -> PipelineResult:
    """Full pipeline.  ``client`` overrides ``config`` (tests inject fakes here)."""
    if not str(intent).strip():
        raise ValueError("需求不能为空")
    eco = ecosystem if isinstance(ecosystem, Ecosystem) else get_ecosystem(ecosystem)

    if client is None:
        cfg = config or resolve_config()
        if cfg is None:
            raise RuntimeError(
                "未配置模型。三种方式任选：\n"
                "  1. 环境变量 OLIVIA_BASE_URL / OLIVIA_MODEL（可选 OLIVIA_API_KEY）\n"
                "  2. 配置文件 ~/.olivia-prompt/config.json "
                '{"base_url": "http://127.0.0.1:8081/v1", "model": "..."}\n'
                "  3. 命令行 --base-url / --model\n"
                "本地 GGUF：llama-server -m models/xxx.gguf --port 8081 后用上面地址。"
            )
        client = ChatClient(cfg)

    fill = fill_slots(intent, client, tier=tier, max_rounds=max_rounds, compact=compact)
    slots = fill.slots
    issues = validate(slots)

    texts = render_all(slots, eco, skeleton=skeleton)
    ne = build_negative(slots, eco)
    prompts = {
        t: {
            "positive": text,
            "negative": ne.text,
            "negative_strength": ne.strength,
            "negative_note": ne.note,
            "positive_patch": list(ne.positive_patch),
        }
        for t, text in texts.items()
    }

    notes = list(fill.feedback)
    if fill.needed_repair:
        notes.append(f"模型经过了 {fill.rounds - 1} 轮修复")
    if eco.strength.value == "none":
        notes.append("该生态负向无效，抑制意图已转成正向约束补丁")

    return PipelineResult(
        intent=str(intent),
        tier=tier,
        slots=slots,
        issues=issues,
        prompts=prompts,
        fill=fill,
        notes=notes,
    )


__all__ = ["PipelineResult", "run_intent"]
