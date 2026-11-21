"""Rule-based batch sampler — M1.

Turns the scene library x persona/activity/style pools into complete slot
payloads *without any model*.  This is the "规则版" the plan calls for: it
proves the skeleton can cover hundreds of distinct worlds and exposes coverage
holes before any training data is spent.

Determinism contract: ``sample_batch(n, seed=k)`` always returns the same
records, and the first ``m`` records of ``sample_batch(n, seed=k)`` for
``m <= n`` are identical too — each record derives its RNG from
``f"{seed}:{index}"``, never from global state.

Provenance: the sampler marks every filled slot ``inferred`` (nothing came
from a real user).  The M4 intent synthesizer overrides the slots it actually
verbalizes to ``user``, so the training targets teach the src distinction.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .pools import (
    ACCESSORIES,
    AGES,
    ASPECTS,
    BARTWEARS,
    CAM_HEIGHTS,
    CRAFTS,
    EXPRESSIONS,
    GRIPS,
    HAIR_LENGTHS,
    MEDIUMS,
    MODEL_STYLES,
    OUTFITS,
    OUTFIT_DETAILS,
    OUTFIT_LAYERS,
    PERSONAS,
    QUALITY,
    SKINS,
    light_palette,
    pick_weighted,
)
from .scenes import ACTIVITIES, SCENE_GROUPS, Scene, activities_for, scenes_for_group
from .slots import SLOT_ORDER, SLOTS

#: plan rule: 同一「场景 × 动作」组合最多出现 3 次
MAX_SCENE_ACTIVITY_USES = 3

#: activities that stay honest while the subject is mid-stride
WALK_OK = frozenset({"map", "photograph", "smoke"})
#: activities that make sense lying down
LIE_OK = frozenset({"read"})

#: Closed-domain slots: their value space is intentionally tiny (booleans,
#: pose-derived camera, palette-derived colors, renderer-injected quality...).
#: They are *reported* in the matrix but never flagged — <15 distinct values is
#: the complete domain, not a data gap.
CLOSED_DOMAIN_SLOTS = frozenset({
    "subject_count", "gender", "age", "hair_length", "indoor",
    "cam_height", "aspect", "medium", "model_style",
    "quality", "render", "cool_color", "warm_color",
})


@dataclass
class SampleRecord:
    id: str
    slots: Dict[str, object]
    meta: Dict[str, object] = field(default_factory=dict)

    def as_row(self) -> dict:
        return {"id": self.id, "meta": self.meta, "slots": self.slots}


def _persona(rng) -> dict:
    hair, hair_zh, eye = rng.choice(PERSONAS)
    skin = rng.choice(SKINS)
    age = rng.choice(AGES)
    expr = rng.choice(EXPRESSIONS)
    length = rng.choice(HAIR_LENGTHS)
    return {
        "hair_color": hair, "hair_color_zh": hair_zh, "eye_color": eye,
        "skin": skin[0], "skin_zh": skin[1],
        "age": age[0], "age_zh": age[1],
        "expression": expr[0], "expression_zh": expr[1],
        "hair_length": length[0], "hair_length_zh": length[1],
    }


def _wardrobe(rng) -> dict:
    main, main_zh = rng.choice(OUTFITS)
    detail, detail_zh = rng.choice(OUTFIT_DETAILS)
    layer, layer_zh = rng.choice(OUTFIT_LAYERS)
    bart, bart_zh = rng.choice(BARTWEARS)
    acc, acc_zh = rng.choice(ACCESSORIES)
    return {
        "outfit_main": main, "outfit_main_zh": main_zh,
        "outfit_detail": detail, "outfit_detail_zh": detail_zh,
        "outfit_layer": layer, "outfit_layer_zh": layer_zh,
        "bartwear": bart, "bartwear_zh": bart_zh,
        "accessories": [acc] if acc else [],
        "accessories_zh": acc_zh,
    }


def _hit_items(rng, hair_color: str, outfit_main: str) -> List[str]:
    garment = outfit_main.split()[-1]
    part = rng.choice(("collar", "sleeve", "hem", "shoulder seam"))
    # no leading article: P4a renders "catching the {hit}"
    return [f"edges of her {hair_color} hair", f"{part} of the {garment}"]


def build_slots(
    scene: Scene,
    activity: dict,
    pose: str,
    rng: random.Random,
    *,
    subject_count: int = 1,
    gender: str = "female",
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Return (slot_payload, meta) for one decided world."""
    persona = _persona(rng)
    wardrobe = _wardrobe(rng)
    cool, warm = light_palette(scene.light_time)
    aspect, aspect_zh = rng.choice(ASPECTS)
    cam_en, cam_zh = CAM_HEIGHTS[pose]
    medium = pick_weighted(rng, [(m, w) for m, w, _ in MEDIUMS])
    render = next(r for m, _, r in MEDIUMS if m == medium)
    light_dir = rng.choice(scene.light_dirs)
    perch = rng.choice(scene.perch_options(pose))
    craft = rng.sample(CRAFTS, 2)
    mood = rng.choice(scene.moods)
    outfit_main_garment = wardrobe["outfit_main"].split()[-1]

    slots: Dict[str, object] = {
        # identity
        "subject_count": subject_count,
        "gender": gender,
        "age": persona["age"],
        "skin": persona["skin"],
        "hair_length": persona["hair_length"],
        "hair_color": persona["hair_color"],
        "eye_color": persona["eye_color"],
        "expression": persona["expression"],
        # wardrobe
        "outfit_main": wardrobe["outfit_main"],
        "outfit_detail": wardrobe["outfit_detail"],
        "outfit_layer": wardrobe["outfit_layer"],
        "bartwear": wardrobe["bartwear"],
        "accessories": wardrobe["accessories"],
        "craft": list(craft),
        # pose / action
        "pose_family": pose,
        "perch": perch,
        "setting": scene.setting,
        "action": activity["action"],
        "handheld": activity["handheld"],
        "hand_support": activity["hand_support"],
        "hand_other": activity["hand_other"],
        "grip": rng.choice(GRIPS),
        "gaze_target": activity["gaze"],
        "visibility": list(activity["visibility"]),
        "ground": scene.ground,
        # world
        "indoor": scene.indoor,
        "props": list(scene.props),
        "anchor": scene.anchor,
        "opening": scene.opening,
        "background": scene.background,
        "bg_far": scene.bg_far,
        "materials": list(scene.materials),
        # light
        "light_time": scene.light_time,
        "light_dir": light_dir,
        "hit": _hit_items(rng, persona["hair_color"], outfit_main_garment),
        "cool_color": cool,
        "warm_color": warm,
        # camera
        "aspect": aspect,
        "cam_height": cam_en,
        # style
        "medium": medium,
        "model_style": rng.choice(MODEL_STYLES),
        "quality": list(QUALITY),
        "render": list(render),
        "mood": mood,
    }

    meta = {
        "scene": scene.id,
        "name_zh": scene.name_zh,
        "group": scene.group,
        "activity": activity["key"],
        "activity_zh": activity["zh"],
        "pose": pose,
        "perch": perch,
        "medium": medium,
        "aspect": aspect,
        "aspect_zh": aspect_zh,
        "cam_height": cam_en,
        "cam_height_zh": cam_zh,
        "light_time": scene.light_time,
        "light_zh": scene.light_zh,
        "light_dir": light_dir,
        "light_dir_zh": None,
        "mood": mood,
        "subject_count": subject_count,
        "gender": gender,
        "persona": persona,
        "wardrobe": wardrobe,
    }
    return slots, meta


def sample_batch(
    n: int,
    seed: int = 0,
    groups: Optional[List[str]] = None,
    gender_ratio: float = 0.1,
) -> List[SampleRecord]:
    """Sample ``n`` records; deterministic for a given (n, seed, groups).

    Enforces the plan's dedup rule: a (scene, activity) pair appears at most
    ``MAX_SCENE_ACTIVITY_USES`` times.  If ``n`` exceeds capacity the batch is
    capped at capacity (60 scenes x ~20 activities x 3 uses).
    """
    picked: List[Scene] = []
    for g in (groups or []):
        picked.extend(scenes_for_group(g))
    pool = picked if groups else list(scenes_for_group(None))
    if not pool:
        raise ValueError(f"no scenes for groups {groups!r}; known: {SCENE_GROUPS}")

    usage: Counter = Counter()
    capacity = sum(len(activities_for(s)) for s in pool) * MAX_SCENE_ACTIVITY_USES
    n = min(n, capacity)

    records: List[SampleRecord] = []
    for i in range(n):
        rng = random.Random(f"{seed}:{i}")
        # rejection-sample a fresh (scene, activity) until under the dedup cap
        for _attempt in range(200):
            scene = rng.choice(pool)
            acts = activities_for(scene)
            activity = rng.choice(acts)
            if usage[(scene.id, activity["key"])] < MAX_SCENE_ACTIVITY_USES:
                break
        else:  # pragma: no cover — capacity is checked above
            break
        usage[(scene.id, activity["key"])] += 1

        pose = rng.choice(scene.fit_poses)
        gender = "male" if rng.random() < gender_ratio else "female"
        # physical honesty: some activities cannot happen mid-stride or lying down
        allowed = [p for p in scene.fit_poses
                   if not (p == "walk" and activity["key"] not in WALK_OK)
                   and not (p == "lie" and activity["key"] not in LIE_OK)]
        if not allowed:
            allowed = [p for p in scene.fit_poses if p != "lie"] or scene.fit_poses
        pose = rng.choice(allowed)
        slots, meta = build_slots(scene, activity, pose, rng, gender=gender)
        meta["light_dir_zh"] = meta["light_dir"]
        records.append(SampleRecord(
            id=f"{scene.id}_{activity['key']}_{pose}_{i:04d}",
            slots=slots,
            meta=meta,
        ))
    return records


# --------------------------------------------------------------------------- #
# coverage matrix — the plan's pre-training gate
# --------------------------------------------------------------------------- #

def coverage_matrix(records: List[SampleRecord]) -> Dict[str, Counter]:
    """Slot -> Counter(values) over the sampled records."""
    matrix: Dict[str, Counter] = {}
    for rec in records:
        for name, value in rec.slots.items():
            if name not in SLOTS:
                continue
            kind = SLOTS[name].kind
            key = tuple(value) if isinstance(value, (list, tuple)) else str(value)
            matrix.setdefault(name, Counter())[key] += 1
    return matrix


def coverage_report(records: List[SampleRecord], min_values: int = 15) -> Tuple[str, List[str]]:
    """Human-readable matrix + the list of slots below ``min_values``.

    The plan's rule ("任一槽独立取值 < 15 种就补数据") only makes sense for
    free-value slots; enum slots are structurally capped (light_dir has 6
    options no matter how much data you synthesize), so they are reported but
    never flagged.
    """
    matrix = coverage_matrix(records)
    lines = [
        f"{'slot':<16} {'distinct':>8}  distribution (top 8)",
        "-" * 88,
    ]
    flagged: List[str] = []
    for name in SLOT_ORDER:
        if name not in matrix:
            continue
        counter = matrix[name]
        slot = SLOTS[name]
        top = ", ".join(f"{v[:24]}:{c}" for v, c in counter.most_common(8))
        lines.append(f"{name:<16} {len(counter):>8}  {top}")
        is_enum = slot.kind == "enum" and slot.options
        closed = name in CLOSED_DOMAIN_SLOTS
        if not is_enum and not closed and len(counter) < min_values:
            flagged.append(name)
    lines.append("-" * 88)
    if flagged:
        lines.append(f"低于 {min_values} 种独立取值（补数据候选）：{', '.join(flagged)}")
    else:
        lines.append(f"所有自由值槽位独立取值均 ≥ {min_values}。")
    return "\n".join(lines), flagged


__all__ = [
    "SampleRecord",
    "build_slots",
    "sample_batch",
    "coverage_matrix",
    "coverage_report",
    "MAX_SCENE_ACTIVITY_USES",
]
