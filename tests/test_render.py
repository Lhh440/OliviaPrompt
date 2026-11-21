"""M0 test suite — stdlib only, runs with or without pytest."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from olivia_prompt import (  # noqa: E402
    SLOTS,
    ECOSYSTEMS,
    SlotSet,
    build_negative,
    render,
    render_all,
    validate,
)
from olivia_prompt.slots import SlotError  # noqa: E402

EXAMPLES = ROOT / "examples"


def load(name: str) -> dict:
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


#: The nine invariant sentence positions of the reference skeleton.
LONG_MARKERS = (
    "A highly detailed anime illustration of a single fair-skinned adult woman",
    "with long silver hair, blue-gray eyes and a calm, attentive expression",
    "She wears a deep navy dress with delicate ivory embroidery",
    "give her appearance a natural, carefully observed quality",
    "relaxed, believable grip",
    "clearly visible",
    "frame the bench",
    "opens onto layers of distant rooftops",
    "progressively softer detail toward the distant city",
    "sunlight enters from the left",
    "shadows balance the amber highlights",
    "composition at seated eye level",
    "create a peaceful, coherent illustration",
)


class TestLongTier(unittest.TestCase):
    def setUp(self):
        self.ss = SlotSet.from_mapping(load("slots_rooftop.json"), tier="long")
        self.text = render(self.ss, "long", "illustrious")

    def test_reproduces_reference_skeleton(self):
        for marker in LONG_MARKERS:
            self.assertIn(marker, self.text, f"missing skeleton marker: {marker}")

    def test_four_body_paragraphs(self):
        blocks = [b for b in self.text.split("\n\n") if b.strip()]
        # prefix line + 4 body paragraphs
        self.assertEqual(len(blocks), 5, blocks)

    def test_prefix_line_shape(self):
        first = self.text.split("\n\n")[0]
        self.assertTrue(first.startswith("masterpiece, very aesthetic,"))
        self.assertTrue(first.endswith(","))
        self.assertEqual(len(first.rstrip(",").split(",")), 4)

    def test_deterministic(self):
        again = render(SlotSet.from_mapping(load("slots_rooftop.json"), tier="long"),
                       "long", "illustrious")
        self.assertEqual(self.text, again)

    def test_gender_pronouns_follow_slot(self):
        raw = load("slots_rooftop.json")
        raw["gender"] = "male"
        ss = SlotSet.from_mapping(raw, tier="long")
        text = render(ss, "long", "illustrious")
        self.assertIn("adult man", text)
        self.assertIn("He wears", text)
        self.assertIn("his appearance", text)
        self.assertNotIn("She wears", text)


class TestPoseFamilies(unittest.TestCase):
    def test_every_pose_renders_and_matches_grammar(self):
        verb_prep = {
            "sit": "sits on", "stand": "stands at", "walk": "walks through",
            "lean": "leans against", "kneel": "kneels on", "lie": "lies on",
            "crouch": "crouches in",
        }
        for pose, phrase in verb_prep.items():
            raw = load("slots_rooftop.json")
            raw["pose_family"] = pose
            ss = SlotSet.from_mapping(raw, tier="long")
            text = render(ss, "long", "illustrious")
            self.assertIn(phrase, text, f"pose {pose} did not render {phrase!r}")

    def test_ground_clause_varies_by_pose(self):
        raw = load("slots_rooftop.json")
        raw["pose_family"] = "kneel"
        ss = SlotSet.from_mapping(raw, tier="long")
        self.assertIn("one knee rests on the stone paving",
                      render(ss, "long", "illustrious"))


class TestMediumAndShort(unittest.TestCase):
    def setUp(self):
        self.ss = SlotSet.from_mapping(load("slots_rooftop.json"), tier="long")

    def test_medium_three_body_paragraphs(self):
        text = render(self.ss, "medium", "illustrious")
        blocks = [b for b in text.split("\n\n") if b.strip()]
        self.assertEqual(len(blocks), 4)

    def test_short_is_clean_tags(self):
        text = render(self.ss, "short", "illustrious")
        self.assertIn("1girl", text)
        self.assertIn("solo", text)
        self.assertIn("silver_hair", text)
        self.assertNotIn(".", text)
        self.assertNotIn("  ", text)
        tags = text.split(", ")
        self.assertEqual(len(tags), len(set(tags)), "duplicate tags")

    def test_short_midjourney_uses_spaces(self):
        text = render(self.ss, "short", "midjourney")
        self.assertNotIn("_", text)

    def test_all_three_tiers(self):
        out = render_all(self.ss, "illustrious")
        self.assertEqual(set(out), {"short", "medium", "long"})


class TestNegative(unittest.TestCase):
    def setUp(self):
        self.ss = SlotSet.from_mapping(load("slots_rooftop.json"), tier="long")

    def test_strong_ecosystem_gets_text(self):
        ne = build_negative(self.ss, "illustrious")
        self.assertEqual(ne.strength, "strong")
        self.assertTrue(ne.text)
        self.assertLessEqual(len(ne.text.split()), 60)

    def test_cfg1_ecosystem_leaves_text_empty_and_patches_positive(self):
        ne = build_negative(self.ss, "krea_turbo")
        self.assertEqual(ne.strength, "none")
        self.assertEqual(ne.text, "")
        self.assertTrue(ne.positive_patch)
        self.assertIn("a single subject only", ne.positive_patch)

    def test_midjourney_becomes_param(self):
        ne = build_negative(self.ss, "midjourney")
        self.assertEqual(ne.strength, "param")
        self.assertTrue(ne.text.startswith("--no "))

    def test_mirrored_from_positive(self):
        ne = build_negative(self.ss, "illustrious")
        joined = ne.text
        # single subject + handheld + seated + visibility + background + light
        self.assertIn("multiple people", joined)
        self.assertIn("object merging into hand", joined)
        self.assertIn("cropped face", joined)
        self.assertIn("cluttered background", joined)
        self.assertIn("flat lighting", joined)

    def test_never_excludes_other_values(self):
        """silver hair must not put blonde/brown hair in the negative."""
        ne = build_negative(self.ss, "illustrious")
        for forbidden in ("blonde hair", "brown hair", "black hair", "white hair"):
            self.assertNotIn(forbidden, ne.text)
        self.assertIn("multicolored hair", ne.text)

    def test_style_exclusion_flips_with_medium(self):
        raw = load("slots_rooftop.json")
        raw["medium"] = "cinematic photography"
        ss = SlotSet.from_mapping(raw, tier="long")
        ne = build_negative(ss, "illustrious")
        self.assertIn("anime", ne.text)
        self.assertNotIn("photorealistic", ne.text)

    def test_pony_uses_score_tokens(self):
        ne = build_negative(self.ss, "pony")
        self.assertIn("score_4", ne.text)

    def test_weak_ecosystem_is_compressed(self):
        ne = build_negative(self.ss, "qwen_image")
        self.assertEqual(ne.strength, "weak")
        self.assertLessEqual(len(ne.text.split()), 18)

    def test_all_ecosystems_build(self):
        for key in ECOSYSTEMS:
            ne = build_negative(self.ss, key)
            self.assertIsInstance(ne.text, str)


class TestValidation(unittest.TestCase):
    def test_clean_payload_has_no_errors(self):
        ss = SlotSet.from_mapping(load("slots_rooftop.json"), tier="long")
        issues = validate(ss)
        self.assertFalse([i for i in issues if i.severity == "error"], issues)

    def test_indoor_with_leaves_warns(self):
        raw = load("slots_rooftop.json")
        raw["indoor"] = True
        ss = SlotSet.from_mapping(raw, tier="long")
        codes = {i.code for i in validate(ss)}
        self.assertIn("W_INDOOR_MATERIAL", codes)

    def test_standing_with_seated_eye_level_errors(self):
        raw = load("slots_rooftop.json")
        raw["pose_family"] = "stand"
        ss = SlotSet.from_mapping(raw, tier="long")
        codes = {i.code for i in validate(ss)}
        self.assertIn("E_CAM_POSE", codes)

    def test_seated_pose_without_ground_errors(self):
        raw = load("slots_rooftop.json")
        raw["ground"] = ""
        ss = SlotSet.from_mapping(raw, tier="long")
        codes = {i.code for i in validate(ss)}
        self.assertIn("E_POSE_GROUND", codes)


class TestSlotSet(unittest.TestCase):
    def test_defaults_render_without_input(self):
        ss = SlotSet.defaults("long")
        text = render(ss, "long", "illustrious")
        self.assertIn("A highly detailed", text)
        self.assertGreater(len(text.split()), 200)

    def test_source_tracking(self):
        raw = load("slots_rooftop.json")
        ss = SlotSet.from_mapping(raw, tier="long")
        self.assertEqual(ss.source("hair_color"), "user")
        self.assertEqual(ss.source("hand_support"), "inferred")
        self.assertEqual(ss.source("tags_extra"), "default")

    def test_open_questions_only_high_impact_and_default(self):
        ss = SlotSet.defaults("long")
        names = {s.name for s in ss.open_questions()}
        for name in names:
            self.assertEqual(SLOTS[name].impact, "high")
            self.assertTrue(ss.is_default(name))
        self.assertIn("light_dir", names)
        self.assertNotIn("mood", names)

    def test_unknown_slot_rejected(self):
        with self.assertRaises(SlotError):
            SlotSet.from_mapping({"not_a_slot": 1})

    def test_bad_enum_rejected(self):
        with self.assertRaises(SlotError):
            SlotSet.from_mapping({"pose_family": "float"})

    def test_list_length_enforced(self):
        with self.assertRaises(SlotError):
            SlotSet.from_mapping({"props": ["only one"]})

    def test_roundtrip_dict(self):
        ss = SlotSet.from_mapping(load("slots_rooftop.json"), tier="long")
        again = SlotSet.from_mapping(ss.as_dict(), tier="long")
        self.assertEqual(render(ss, "long", "illustrious"),
                         render(again, "long", "illustrious"))


class TestExamples(unittest.TestCase):
    def test_cyberpunk_example_is_consistent(self):
        ss = SlotSet.from_mapping(load("slots_cyberpunk.json"), tier="long")
        errors = [i for i in validate(ss) if i.severity == "error"]
        self.assertFalse(errors, errors)
        text = render(ss, "long", "illustrious")
        self.assertIn("cinematic photography", text)
        self.assertIn("photograph.", text)  # medium noun in P4d

    def test_cyberpunk_negative_excludes_anime(self):
        ss = SlotSet.from_mapping(load("slots_cyberpunk.json"), tier="long")
        ne = build_negative(ss, "illustrious")
        self.assertIn("anime", ne.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
