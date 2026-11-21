"""Long-tier skeleton variants.

The default long skeleton ("A") is the reference four-paragraph / fourteen
sentence-position layout that M0 pinned down.  Per the plan (开工前确认 #2) and
the user's 骨架 notes, three further variants exist so a user can fork their own
style without retraining anything:

    A  参考骨架          prefix → P1 → P2 → P3 → P4           (default)
    B  换段序            prefix → P1 → P3 → P2 → P4           (world before pose)
    C  换判词句          same order as A, different judgement clauses
    D  换 P3 结构        P3 rebuilt depth-first (opening → props → materials)

Every variant consumes the same slot table and the same sentence builders, so a
variant only re-arranges/re-words — it cannot invent new information.  The same
SlotSet + same variant always renders byte-identically.

These variants are opt-in: ``render(slots, "long", skeleton="B")`` or
``olivia-prompt render --skeleton B``.  Default output never changes.
"""

from __future__ import annotations

from typing import Dict

from .slots import SlotSet

# orderings over the blocks returned by render._long_parts
_ORDERS: Dict[str, tuple] = {
    "A": ("prefix", "p1", "p2", "p3", "p4"),
    "B": ("prefix", "p1", "p3", "p2", "p4"),
    "C": ("prefix", "p1", "p2", "p3", "p4"),
    "D": ("prefix", "p1", "p2", "p3", "p4"),
}

#: how each variant feeds the paragraph builders
_PARAMS: Dict[str, dict] = {
    "A": {"judge": "A", "world": "A"},
    "B": {"judge": "A", "world": "A"},
    "C": {"judge": "C", "world": "A"},
    "D": {"judge": "A", "world": "D"},
}

SKELETONS: tuple = ("A", "B", "C", "D")
DEFAULT_SKELETON = "A"

#: one-line human description per variant (CLI / web UI)
SKELETON_DOCS: Dict[str, str] = {
    "A": "参考骨架：主体 → 动作 → 环境 → 光色（默认）",
    "B": "换段序：主体 → 环境 → 动作 → 光色（先立世界再立人）",
    "C": "换判词句：同段序，判词句族替换（unhurried care / nothing out of frame…）",
    "D": "换 P3 结构：环境段改为纵深优先（开口 → 前景 → 材质）",
}


def build_long(ss: SlotSet, skeleton: str, parts_factory) -> str:
    """Compose a long-tier prompt for the given skeleton.

    ``parts_factory`` is :func:`olivia_prompt.render._long_parts` — passed in to
    avoid a circular import.
    """
    if skeleton not in _ORDERS:
        raise ValueError(f"unknown skeleton {skeleton!r}; choose from {SKELETONS}")
    parts = parts_factory(ss, **_PARAMS[skeleton])
    blocks = [parts[k] for k in _ORDERS[skeleton] if parts[k]]
    return "\n\n".join(blocks)


__all__ = [
    "SKELETONS",
    "DEFAULT_SKELETON",
    "SKELETON_DOCS",
    "build_long",
]
