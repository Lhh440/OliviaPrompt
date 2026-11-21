"""Scene library — the world side of sampling.

Every scene is one *place* with the fields the long-tier skeleton needs to
describe a world: framing props, the background opening, what lies beyond it,
material triads (ordered to fit the P3c sentence slots), the natural light
window and which poses are physically honest here.

The data lives in ``data/scenes.json`` (translated from the user's 60-scene
library); this module only loads, validates and indexes it.  Scenes are inputs
to :mod:`olivia_prompt.sample`, never rendered directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

_DATA = Path(__file__).resolve().parent / "data"

VALID_POSES = ("sit", "stand", "walk", "lean", "kneel", "lie", "crouch")
#: every renderer sentence that touches a perch needs one for that pose
POSES_NEEDING_PERCH = VALID_POSES
VALID_LIGHT_DIRS = ("left", "right", "front", "top", "back-right", "back-left")


@dataclass(frozen=True)
class Scene:
    id: str
    name_zh: str
    name_en: str
    group: str
    zh_place: str
    setting: str
    indoor: bool
    props: Tuple[str, str, str]
    opening: str
    background: str
    bg_far: str
    materials: Tuple[str, str, str]
    light_zh: str
    light_time: str
    light_dirs: Tuple[str, ...]
    fit_poses: Tuple[str, ...]
    perches: Dict[str, Tuple[str, ...]]
    ground: str
    anchor: str
    moods: Tuple[str, ...]

    def perch_options(self, pose: str) -> Tuple[str, ...]:
        return self.perches.get(pose, ())


def _require(cond: bool, scene_id: str, what: str) -> None:
    if not cond:
        raise ValueError(f"scene {scene_id!r}: {what}")


def _build_scene(raw: dict) -> Scene:
    sid = str(raw.get("id") or "")
    perches_raw = raw.get("perches") or {}
    poses = tuple(raw.get("fit_poses") or ())
    perches = {p: tuple(opts) for p, opts in perches_raw.items()}

    _require(bool(sid), sid, "missing id")
    _require(bool(raw.get("setting")), sid, "missing setting")
    _require(isinstance(raw.get("indoor"), bool), sid, "indoor must be a bool")
    _require(len(raw.get("props") or ()) == 3, sid, "props must have exactly 3 items")
    _require(bool(raw.get("opening")), sid, "missing opening")
    _require(bool(raw.get("background")), sid, "missing background")
    _require(bool(raw.get("bg_far")), sid, "missing bg_far")
    _require(len(raw.get("materials") or ()) == 3, sid, "materials must have exactly 3 items")
    _require(bool(raw.get("light_time")), sid, "missing light_time")
    _require(bool(poses), sid, "fit_poses must be non-empty")
    for p in poses:
        _require(p in VALID_POSES, sid, f"unknown fit pose {p!r}")
    for p in poses:
        _require(len(perches.get(p, ())) >= 1, sid, f"pose {p!r} needs at least one perch")
        for opts in perches.values():
            _require(all(isinstance(o, str) and o for o in opts), sid, "perch options must be non-empty strings")
    for d in raw.get("light_dirs") or ():
        _require(d in VALID_LIGHT_DIRS, sid, f"unknown light_dir {d!r}")
    _require(bool(raw.get("ground")), sid, "missing ground (seated/grounded poses need it)")
    _require(bool(raw.get("anchor")), sid, "missing anchor")
    _require(bool(raw.get("moods")), sid, "moods must be non-empty")
    _require(bool(raw.get("zh_place")), sid, "missing zh_place")
    _require(bool(raw.get("light_zh")), sid, "missing light_zh")

    return Scene(
        id=sid,
        name_zh=str(raw.get("name_zh") or ""),
        name_en=str(raw.get("name_en") or ""),
        group=str(raw.get("group") or ""),
        zh_place=str(raw.get("zh_place") or ""),
        setting=str(raw["setting"]),
        indoor=bool(raw["indoor"]),
        props=tuple(raw["props"]),
        opening=str(raw["opening"]),
        background=str(raw["background"]),
        bg_far=str(raw["bg_far"]),
        materials=tuple(raw["materials"]),
        light_zh=str(raw.get("light_zh") or ""),
        light_time=str(raw["light_time"]),
        light_dirs=tuple(raw.get("light_dirs") or VALID_LIGHT_DIRS),
        fit_poses=poses,
        perches=perches,
        ground=str(raw["ground"]),
        anchor=str(raw["anchor"]),
        moods=tuple(raw["moods"]),
    )


def _load(path: Path) -> list:
    raw = json.loads(path.read_text(encoding="utf-8"))
    scenes = [_build_scene(s) for s in raw["scenes"]]
    ids = [s.id for s in scenes]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate scene ids: {sorted(dupes)}")
    return scenes


def _load_activities(path: Path) -> List[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: List[dict] = []
    for a in raw["activities"]:
        for field in ("key", "zh", "action", "handheld", "hand_support", "hand_other", "gaze"):
            if not a.get(field):
                raise ValueError(f"activity {a.get('key')!r}: missing {field}")
        if len(a.get("visibility") or ()) != 3:
            raise ValueError(f"activity {a.get('key')!r}: visibility must have exactly 3 items")
        out.append(a)
    keys = [a["key"] for a in out]
    dupes = {k for k in keys if keys.count(k) > 1}
    if dupes:
        raise ValueError(f"duplicate activity keys: {sorted(dupes)}")
    return out


SCENES: Tuple[Scene, ...] = tuple(_load(_DATA / "scenes.json"))
ACTIVITIES: Tuple[dict, ...] = tuple(_load_activities(_DATA / "activities.json"))

SCENE_GROUPS: Tuple[str, ...] = tuple(dict.fromkeys(s.group for s in SCENES))
SCENES_BY_ID: Dict[str, Scene] = {s.id: s for s in SCENES}


def scenes_for_group(group: str | None) -> List[Scene]:
    if not group or group == "all":
        return list(SCENES)
    return [s for s in SCENES if s.group == group]


def activities_for(scene: Scene) -> List[dict]:
    """Activities that make physical sense in this scene."""
    return [a for a in ACTIVITIES if not a.get("only") or scene.id in a["only"]]


def scene_brief(s: Scene) -> dict:
    """Compact metadata for manifests and the web UI."""
    return {
        "id": s.id,
        "name_zh": s.name_zh,
        "name_en": s.name_en,
        "group": s.group,
        "indoor": s.indoor,
        "fit_poses": list(s.fit_poses),
        "light_zh": s.light_zh,
    }


__all__ = [
    "Scene",
    "SCENES",
    "SCENES_BY_ID",
    "SCENE_GROUPS",
    "ACTIVITIES",
    "VALID_POSES",
    "POSES_NEEDING_PERCH",
    "VALID_LIGHT_DIRS",
    "scenes_for_group",
    "activities_for",
    "scene_brief",
]
