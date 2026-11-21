"""Self-consistency rules.

Slots are decided independently by the model, which means they can contradict
each other: a seated pose with no ground plane, an indoor scene described with
leaf veins, a "seated eye level" camera on a standing subject.  A language model
will not reliably notice this; twenty lines of rules will.

Two severities:
    error  — the prompt will visibly break; refuse to render silently
    warn   — it will render, but something is off
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .slots import SlotSet

_SEATED = {"sit", "kneel", "lie", "lean", "crouch"}
_OUTDOOR_MATERIALS = ("leaf", "leaves", "ivy", "grass", "vine", "foliage", "bark", "branch")
# marble is deliberately absent: marble temples, statues and ruins are the
# most common *outdoor* stone, so flagging it would fire on honest scenes.
_INDOOR_MATERIALS = ("paper", "plaster", "tile", "carpet", "linen", "bookshelf")


@dataclass(frozen=True)
class Issue:
    code: str
    severity: str  # "error" | "warn"
    slot: str
    message: str

    def __str__(self) -> str:
        mark = "ERROR" if self.severity == "error" else "WARN "
        return f"[{mark}] {self.code} ({self.slot}): {self.message}"


def validate(slots: SlotSet) -> List[Issue]:
    """Return every self-consistency problem in the slot table."""
    issues: List[Issue] = []
    pose = str(slots["pose_family"])
    indoor = bool(slots["indoor"])
    ground = (slots["ground"] or "").strip()
    materials = [m.lower() for m in (slots.get("materials") or ())]

    # --- ground plane ----------------------------------------------------
    if pose in _SEATED and not ground:
        issues.append(Issue(
            "E_POSE_GROUND", "error", "ground",
            f"姿势族 {pose} 需要 ground（地面材质），否则长档 P2b 的落地从句无内容。",
        ))

    # --- camera height vs pose -------------------------------------------
    cam = str(slots["cam_height"]).lower()
    if pose in ("stand", "walk") and "seated" in cam:
        issues.append(Issue(
            "E_CAM_POSE", "error", "cam_height",
            f"cam_height={slots['cam_height']!r} 与姿势族 {pose} 冲突：站立/行走不能是坐姿视平线。",
        ))
    if pose in ("lie", "crouch") and "eye level" in cam and "seated" in cam:
        issues.append(Issue(
            "E_CAM_POSE_LOW", "error", "cam_height",
            f"cam_height={slots['cam_height']!r} 与姿势族 {pose} 冲突：躺/蹲不适用坐姿视平线。",
        ))

    # --- indoor vs outdoor materials -------------------------------------
    if indoor:
        bad = [m for m in materials if any(o in m for o in _OUTDOOR_MATERIALS)]
        if bad:
            issues.append(Issue(
                "W_INDOOR_MATERIAL", "warn", "materials",
                f"室内场景出现了户外材质 {bad}，P3c 的「叶脉」从句会自相矛盾。",
            ))
    else:
        bad = [m for m in materials if any(i in m for i in _INDOOR_MATERIALS)]
        if bad:
            issues.append(Issue(
                "W_OUTDOOR_MATERIAL", "warn", "materials",
                f"室外场景出现了室内材质 {bad}，请确认 P3c 是否合理。",
            ))

    # --- lighting completeness -------------------------------------------
    if slots["light_time"] and not slots["light_dir"]:
        issues.append(Issue(
            "W_LIGHT_DIR", "warn", "light_dir",
            "给了光照时段但没有光向，P4a/P4b 的阴影方向会缺依据。",
        ))

    # --- multiple subjects ------------------------------------------------
    # (the renderer now pluralises pronouns/verbs from subject_count itself,
    #  so a bare "count > 1" is no longer an inconsistency — only the
    #  single-person visibility directive below still deserves a warning)
    count = str(slots["subject_count"])
    vis = [v for v in (slots.get("visibility") or ()) if v]
    if count != "1" and len(vis) > 1:
        issues.append(Issue(
            "W_VISIBILITY_MULTI", "warn", "visibility",
            "多人场景仍使用单人可见性指令（face/hands/object），建议改写。",
        ))

    # --- handheld vs action ----------------------------------------------
    handheld = str(slots["handheld"]).lower()
    action = str(slots["action"]).lower()
    if handheld and not any(w in action for w in handheld.split()[:2]):
        issues.append(Issue(
            "W_HANDHELD_ACTION", "warn", "action",
            f"handheld={slots['handheld']!r} 未在 action 中提及，P2a 的持物描述可能接不上。",
        ))

    # --- long-tier completeness -------------------------------------------
    if slots.tier == "long":
        for name in ("opening", "anchor", "perch", "bg_far"):
            if slots.is_default(name):
                issues.append(Issue(
                    "W_LONG_DEFAULT", "warn", name,
                    f"长档槽位 {name} 仍是默认值 {slots[name]!r}，输出会显得像模板。",
                ))
        if not any(slots.get(s) for s in ("props", "materials")):
            issues.append(Issue(
                "W_NO_WORLD", "warn", "props",
                "缺少前景道具或材质清单，P3 会退化成通用描述。",
            ))

    # --- quality / mood ----------------------------------------------------
    if not slots["medium"]:
        issues.append(Issue(
            "E_NO_MEDIUM", "error", "medium",
            "medium 为空，前缀行与 P1a 都无法渲染。",
        ))
    if not (slots.get("render") or ()):
        issues.append(Issue(
            "W_NO_RENDER", "warn", "render",
            "缺少渲染描述，P4d 判词句会使用兜底值。",
        ))

    # --- subject count enum sanity ---------------------------------------
    if str(slots["subject_count"]) not in ("1", "2", "3"):
        issues.append(Issue(
            "E_COUNT_RANGE", "error", "subject_count",
            f"subject_count={slots['subject_count']!r} 超出骨架支持范围 1-3。",
        ))

    return issues


def has_errors(issues: List[Issue]) -> bool:
    return any(i.severity == "error" for i in issues)


__all__ = ["Issue", "validate", "has_errors"]
