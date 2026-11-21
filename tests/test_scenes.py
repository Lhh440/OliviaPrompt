"""M1 — scene library integrity."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from olivia_prompt.render import render_all  # noqa: E402
from olivia_prompt.sample import build_slots  # noqa: E402
from olivia_prompt.scenes import (  # noqa: E402
    ACTIVITIES,
    SCENES,
    SCENE_GROUPS,
    SCENES_BY_ID,
    Scene,
    activities_for,
    scenes_for_group,
)
from olivia_prompt.slots import SlotSet  # noqa: E402
from olivia_prompt.validate import validate  # noqa: E402

import random  # noqa: E402


class TestSceneLibrary(unittest.TestCase):
    def test_sixty_scenes_load(self):
        self.assertGreaterEqual(len(SCENES), 60)
        ids = [s.id for s in SCENES]
        self.assertEqual(len(ids), len(set(ids)), "duplicate scene ids")

    def test_groups_partition_all_scenes(self):
        covered = sum(len(scenes_for_group(g)) for g in SCENE_GROUPS)
        self.assertEqual(covered, len(SCENES))

    def test_rooftop_reference_not_required_but_fields_shape_ok(self):
        s = SCENES_BY_ID["study_dawn"]
        self.assertTrue(s.indoor)
        self.assertEqual(len(s.props), 3)
        self.assertEqual(len(s.materials), 3)
        self.assertTrue(s.perch_options("sit"))
        self.assertNotIn("lie", s.fit_poses)

    def test_every_scene_renders_with_every_fit_pose(self):
        """For every (scene, fit pose) pair a full payload must validate clean
        (no error-severity issues) and render in all three tiers."""
        checked = 0
        for scene in SCENES:
            for pose in scene.fit_poses:
                rng = random.Random(f"probe:{scene.id}:{pose}")
                activity = activities_for(scene)[0]
                slots, _meta = build_slots(scene, activity, pose, rng)
                ss = SlotSet.from_mapping(slots, tier="long")
                errors = [i for i in validate(ss) if i.severity == "error"]
                self.assertFalse(errors, f"{scene.id}/{pose}: {errors}")
                for tier, text in render_all(ss).items():
                    self.assertTrue(text.strip(), f"{scene.id}/{pose}/{tier} empty")
                checked += 1
        self.assertGreater(checked, 100)

    def test_no_indoor_outdoor_material_contradictions(self):
        """Scenes are authored data — the W_INDOOR_MATERIAL warn must never
        fire on them, or the sampler floods downstream consumers with noise."""
        for scene in SCENES:
            rng = random.Random(f"probe2:{scene.id}")
            activity = activities_for(scene)[0]
            pose = scene.fit_poses[0]
            slots, _ = build_slots(scene, activity, pose, rng)
            ss = SlotSet.from_mapping(slots, tier="long")
            codes = {i.code for i in validate(ss)}
            self.assertNotIn("W_INDOOR_MATERIAL", codes, scene.id)
            self.assertNotIn("W_OUTDOOR_MATERIAL", codes, scene.id)

    def test_light_time_matches_phase_vocabulary(self):
        """light_time must start capitalized (sentence-initial in P4a) and not
        end with a noun ('...light sunlight enters' would be broken)."""
        bad_endings = ("light", "sunlight", "glow", "rays", "beam", "backlight")
        for scene in SCENES:
            lt = scene.light_time
            self.assertEqual(lt, lt[:1].upper() + lt[1:], scene.id)
            self.assertFalse(lt.lower().endswith(bad_endings), scene.id)

    def test_openings_do_not_start_with_vowel_wrecking_articles(self):
        # the renderer now handles a/an automatically; this only asserts that
        # openings are bare noun phrases (no leading article of their own)
        for scene in SCENES:
            low = scene.opening.lower()
            self.assertFalse(low.startswith(("a ", "an ", "the ")), scene.id)

    def test_activities_shape_and_only_refs_exist(self):
        for a in ACTIVITIES:
            # same semantics as validate.W_HANDHELD_ACTION: any of the first
            # two words must appear in the action phrase
            self.assertTrue(
                any(w in a["action"] for w in a["handheld"].split()[:2]),
                f"activity {a['key']}: handheld words missing from action",
            )
            for sid in a.get("only", []):
                self.assertIn(sid, SCENES_BY_ID, f"activity {a['key']}: unknown scene {sid}")

    def test_every_scene_has_at_least_one_activity(self):
        for scene in SCENES:
            self.assertTrue(activities_for(scene), scene.id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
