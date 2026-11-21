"""Skeleton variants B/C/D."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from olivia_prompt.render import render  # noqa: E402
from olivia_prompt.slots import SlotSet  # noqa: E402
from olivia_prompt.skeletons import SKELETONS, build_long  # noqa: E402

EXAMPLES = ROOT / "examples"


def load(name: str) -> dict:
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


class TestSkeletonVariants(unittest.TestCase):
    def setUp(self):
        self.ss = SlotSet.from_mapping(load("slots_rooftop.json"), tier="long")

    def test_a_is_default_and_matches_reference_markers(self):
        self.assertEqual(render(self.ss, "long"), render(self.ss, "long", skeleton="A"))
        text = render(self.ss, "long", skeleton="A")
        self.assertIn("opens onto layers of distant rooftops", text)
        self.assertIn("give her appearance a natural, carefully observed quality", text)

    def test_all_variants_render_differently(self):
        texts = {sk: render(self.ss, "long", skeleton=sk) for sk in SKELETONS}
        self.assertEqual(len(set(texts.values())), len(SKELETONS))

    def test_variants_deterministic(self):
        for sk in SKELETONS:
            self.assertEqual(
                render(self.ss, "long", skeleton=sk),
                render(SlotSet.from_mapping(load("slots_rooftop.json"), tier="long"),
                       "long", skeleton=sk),
            )

    def test_variants_keep_core_information(self):
        for sk in SKELETONS:
            t = render(self.ss, "long", skeleton=sk)
            markers = (
                "sunlight enters from the left",
                "relaxed, believable grip",
                "composition at seated eye level",
            )
            for marker in markers:
                self.assertIn(marker, t, f"skeleton {sk} lost {marker!r}")
            # visibility directive: A/B/D use the default wording, C its own
            if sk == "C":
                self.assertIn("stay in clear view", t)
            else:
                self.assertIn("clearly visible", t)

    def test_variant_b_world_before_pose(self):
        t = render(self.ss, "long", skeleton="B")
        world_pos = t.index("frame the bench")
        pose_pos = t.index("She sits on a wooden bench")
        self.assertLess(world_pos, pose_pos)

    def test_variant_c_swaps_judgements(self):
        t = render(self.ss, "long", skeleton="C")
        self.assertIn("deliberate, unhurried care", t)
        self.assertIn("stay in clear view", t)
        self.assertNotIn("clearly visible", t)

    def test_variant_d_rebuilds_p3_depth_first(self):
        t = render(self.ss, "long", skeleton="D")
        self.assertIn("Beyond a low wrought-iron railing", t)
        self.assertIn("recede toward the distant city", t)
        opening_pos = t.index("Beyond a low wrought-iron railing")
        props_pos = t.index("frame the bench")
        self.assertLess(opening_pos, props_pos)

    def test_unknown_skeleton_rejected(self):
        with self.assertRaises(ValueError):
            render(self.ss, "long", skeleton="E")
        with self.assertRaises(ValueError):
            build_long(self.ss, "Z", lambda ss: {})

    def test_skeleton_ignored_for_short_and_medium(self):
        short_a = render(self.ss, "short")
        self.assertEqual(short_a, render(self.ss, "short", skeleton="D"))
        medium_a = render(self.ss, "medium")
        self.assertEqual(medium_a, render(self.ss, "medium", skeleton="B"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
