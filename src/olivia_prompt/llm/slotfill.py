"""Intent -> slot JSON: prompt building, parsing, and the repair loop.

The contract with the model is narrow on purpose:

* it sees only the slots visible in the requested tier,
* it must emit a single JSON object ``{"slot_name": {"value": ..., "src": ...}}``,
* ``src`` is a checkable fact (user / inferred / default), not a confidence,
* it never writes prose — the renderers own every sentence.

Anything the model gets wrong (unknown slot, bad enum, self-contradiction) is
fed back verbatim for one or two repair rounds.  That loop is what makes a 4B
model reliable enough to sit in the product path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List

from ..slots import SLOTS, SLOT_ORDER, SlotError, SlotSet
from ..validate import validate as run_validate
from .client import ChatClient

REPAIR_ROUNDS = 2

#: The fine-tuned-model serving contract, used verbatim as the system message
#: at SFT time and at inference time.  Single source of truth lives here (the
#: product package); the training synthesizer imports it — never the reverse.
#: Keep it stable: changing it silently invalidates trained checkpoints.
INSTRUCTION = (
    "你是绘画提示词槽位填充器。读取用户的中文画面需求，输出该档位的完整槽位 JSON："
    '键为槽位名，值为 {"value": …, "src": "user|inferred|default"}；'
    "src 只能是 user（需求明确说了）/ inferred（合理推断）/ default（无信息保留默认）。"
    "只输出 JSON，不要解释。保持自洽：坐跪躺蹲需要 ground，室内不出现树叶苔藓，"
    "机位高度符合姿势族，持有物要在 action 里被提到。"
)

_RULES = """\
规则：
1. 只输出一个 JSON 对象，不要 markdown 代码块，不要解释。
2. 每个键是槽位名，值是 {"value": ..., "src": "..."}；src 只能是
   user（用户明确说了）/ inferred（你合理推断）/ default（无信息，保留默认）。
3. 用户提到的信息标 user，绝不追问；没提到的保持 default。
4. 保持自洽：坐/跪/躺/蹲需要 ground；室内不出现树叶苔藓；机位高度必须
   符合姿势族（站姿不能用坐姿视平线）；持有物必须在 action 里被提到。
5. 槽位值用英文渲染词汇；不要发明列表之外的槽位名。
6. 光照时段 light_time 写成英文句首形容词短语（如 Warm late-afternoon、
   Neon-lit night），不要以名词结尾。"""


def slot_brief(tier: str) -> str:
    lines: List[str] = []
    for name in SLOT_ORDER:
        s = SLOTS[name]
        if not s.in_tier(tier):
            continue
        head = f"{name} ({s.impact}, {s.kind})"
        if s.options:
            head += f" [{'|'.join(s.options)}]"
        default = s.default
        if isinstance(default, tuple):
            default = list(default)
        lines.append(f"- {head} 默认={default!r}；{s.doc}")
    return "\n".join(lines)


def system_prompt(tier: str) -> str:
    few_shot = json.dumps(_FEWSHOT, ensure_ascii=False, indent=1)
    return (
        "你是绘画提示词的槽位填充器。用户给出中文画面需求，"
        f"你为「{tier}」档位输出完整的槽位 JSON。\n\n"
        f"{_RULES}\n\n"
        f"档位可见槽位及默认值：\n{slot_brief(tier)}\n\n"
        f"输出形状示例（内容仅供格式参考）：\n{few_shot}"
    )


_FEWSHOT = {
    "subject_count": {"value": 1, "src": "user"},
    "gender": {"value": "female", "src": "user"},
    "hair_color": {"value": "silver", "src": "user"},
    "setting": {"value": "quiet rooftop garden", "src": "user"},
    "action": {"value": "drawing in an open sketchbook", "src": "user"},
    "pose_family": {"value": "sit", "src": "user"},
    "perch": {"value": "wooden bench", "src": "inferred"},
    "ground": {"value": "stone paving", "src": "inferred"},
    "light_time": {"value": "Warm late-afternoon", "src": "inferred"},
    "light_dir": {"value": "left", "src": "inferred"},
    "cam_height": {"value": "seated eye level", "src": "inferred"},
    "medium": {"value": "anime illustration", "src": "default"},
}


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #

_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def extract_json(text: str) -> dict:
    """Pull the first balanced JSON object out of a model reply."""
    if not text:
        raise SlotError("模型返回为空")
    # reasoning models (qwen3 gguf) may emit a think block first — drop it so
    # a stray "{" inside the thinking cannot hijack the extraction
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    # drop markdown fences if present
    cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    if start < 0:
        raise SlotError(f"模型返回中没有 JSON 对象：{text[:120]!r}")

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = cleaned[start:i + 1]
                return _loads_repaired(blob)
    raise SlotError(f"JSON 对象未闭合：{text[:120]!r}")


def _loads_repaired(blob: str) -> dict:
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        # common 4B-model slips: trailing commas, smart quotes
        fixed = _TRAILING_COMMA.sub(r"\1", blob)
        fixed = fixed.replace("「", '"').replace("」", '"').replace("：", ":")
        try:
            data = json.loads(fixed)
        except json.JSONDecodeError as exc:
            raise SlotError(f"JSON 解析失败：{exc}") from exc
    if not isinstance(data, dict):
        raise SlotError("顶层必须是 JSON 对象")
    return data


def parse_slots(text: str, tier: str) -> SlotSet:
    return SlotSet.from_mapping(extract_json(text), tier=tier)


# --------------------------------------------------------------------------- #
# fill loop
# --------------------------------------------------------------------------- #

@dataclass
class FillResult:
    slots: SlotSet
    raw: str
    rounds: int = 1
    feedback: List[str] = field(default_factory=list)

    @property
    def needed_repair(self) -> bool:
        return self.rounds > 1


def _problems(slots: SlotSet) -> List[str]:
    """Everything worth sending back to the model for a repair round."""
    # unknown slots / bad enums already raise SlotError before we get here;
    # what remains is self-consistency the model can fix by editing values.
    return [str(i) for i in run_validate(slots) if i.severity == "error"]


def fill_slots(
    intent: str,
    client: ChatClient,
    tier: str = "long",
    max_rounds: int = REPAIR_ROUNDS,
    compact: bool = False,
) -> FillResult:
    """Ask, parse, and repair.  Raises SlotError only when rounds are exhausted.

    ``compact=True`` is the fine-tuned-model contract: the exact system/user
    shape seen at SFT time (system=INSTRUCTION, user="[档位:t] intent").
    ``compact=False`` sends the full schema few-shot prompt for generic models.
    """
    system = INSTRUCTION if compact else system_prompt(tier)
    user = f"[档位:{tier}] {intent}".strip() if compact else str(intent).strip()
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    feedback: List[str] = []
    raw = ""
    for round_no in range(1, max_rounds + 2):
        raw = client.chat(messages, json_mode=True)
        try:
            slots = parse_slots(raw, tier)
        except SlotError as exc:
            feedback.append(f"round {round_no}: {exc}")
            if round_no > max_rounds:
                raise
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": (
                    f"你的输出无法解析：{exc}\n"
                    "重新输出完整的槽位 JSON，只输出 JSON 本体。"
                )},
            ]
            continue

        problems = _problems(slots)
        if not problems:
            return FillResult(slots=slots, raw=raw, rounds=round_no, feedback=feedback)

        feedback.extend(f"round {round_no}: {p}" for p in problems)
        if round_no > max_rounds:
            # last round still self-contradictory: return it, caller decides
            return FillResult(slots=slots, raw=raw, rounds=round_no, feedback=feedback)
        messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": (
                "你的槽位表有以下自洽问题：\n- "
                + "\n- ".join(problems)
                + "\n只输出修正后的完整槽位 JSON。"
            )},
        ]

    raise SlotError("修复轮次耗尽")  # pragma: no cover — loop always returns


__all__ = ["FillResult", "fill_slots", "system_prompt", "slot_brief", "extract_json", "parse_slots"]
