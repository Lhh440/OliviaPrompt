"""M1 — rule-based sampler."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from olivia_prompt.negative import build_negative  # noqa: E402
from olivia_prompt.render import render_all  # noqa: E402
from olivia_prompt.sample import (  # noqa: E402
    MAX_SCENE_ACTIVITY_USES,
    coverage_matrix,
    coverage_report,
    sample_batch,
)
from olivia_prompt.slots import SlotSet  # noqa: E402
from olivia_prompt.validate import has_errors, validate  # noqa: E402


class TestSampler(unittest.TestCase):
    def test_batch_size_and_validity(self):
        recs = sample_batch(200, seed=0)
        self.assertEqual(len(recs), 200)
        for r in recs:
            ss = SlotSet.from_mapping(r.slots, tier="long")
            self.assertFalse(has_errors(validate(ss)), f"{r.id}: {validate(ss)}")
            for tier, text in render_all(ss).items():
                self.assertTrue(text.strip())
            ne = build_negative(ss, "illustrious")
            self.assertTrue(ne.text or ne.positive_patch)

    def test_deterministic_and_prefix_stable(self):
        a = sample_batch(120, seed=7)
        b = sample_batch(120, seed=7)
        self.assertEqual([r.as_row() for r in a], [r.as_row() for r in b])
        c = sample_batch(60, seed=7)
        self.assertEqual([r.as_row() for r in a[:60]], [r.as_row() for r in c])

    def test_scene_activity_dedup_cap(self):
        recs = sample_batch(600, seed=1)
        pairs = {}
        for r in recs:
            key = (r.meta["scene"], r.meta["activity"])
            pairs[key] = pairs.get(key, 0) + 1
        self.assertLessEqual(max(pairs.values()), MAX_SCENE_ACTIVITY_USES)

    def test_groups_filter(self):
        recs = sample_batch(50, seed=3, groups=["fantasy"])
        self.assertTrue(recs)
        self.assertTrue(all(r.meta["group"] == "fantasy" for r in recs))

    def test_pose_matches_scene_fit_poses(self):
        for r in sample_batch(150, seed=2):
            from olivia_prompt.scenes import SCENES_BY_ID
            scene = SCENES_BY_ID[r.meta["scene"]]
            self.assertIn(r.meta["pose"], scene.fit_poses)
            self.assertIn(r.slots["perch"], scene.perch_options(r.meta["pose"]))

    def test_all_records_distinct_ids(self):
        recs = sample_batch(300, seed=5)
        ids = [r.id for r in recs]
        self.assertEqual(len(ids), len(set(ids)))

    def test_walk_and_lie_poses_stay_honest(self):
        from olivia_prompt.sample import LIE_OK, WALK_OK
        for r in sample_batch(250, seed=11):
            act, pose = r.meta["activity"], r.meta["pose"]
            if pose == "walk":
                self.assertIn(act, WALK_OK, f"{r.id}: walking while {act}")
            if pose == "lie":
                self.assertIn(act, LIE_OK, f"{r.id}: lying while {act}")


class TestCoverage(unittest.TestCase):
    def test_matrix_counts_every_record(self):
        recs = sample_batch(80, seed=0)
        m = coverage_matrix(recs)
        self.assertEqual(sum(sum(c.values()) for c in m.values()) >= 80, True)
        self.assertIn("setting", m)
        self.assertIn("materials", m)

    def test_report_flags_open_slots_not_closed(self):
        recs = sample_batch(200, seed=0)
        _rep, flagged = coverage_report(recs)
        # closed domains are never flagged
        for closed in ("cam_height", "medium", "quality", "cool_color"):
            self.assertNotIn(closed, flagged)
        # genuinely open slots with small pools are flagged
        self.assertIn("skin", flagged)
        self.assertIn("grip", flagged)

    def test_report_is_readable(self):
        recs = sample_batch(50, seed=0)
        rep, _ = coverage_report(recs)
        self.assertIn("slot", rep.splitlines()[0])
        self.assertGreater(len(rep.splitlines()), 30)


if __name__ == "__main__":
    unittest.main(verbosity=2)
