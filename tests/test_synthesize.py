"""M4 — data synthesizer, benchmark, eval scoring, export helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "src"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from olivia_prompt.sample import sample_batch  # noqa: E402
from olivia_prompt.slots import SlotError, SlotSet  # noqa: E402
from olivia_prompt.validate import has_errors  # noqa: E402
from training.data.synthesize import (  # noqa: E402
    USER_SLOTS,
    build_pair,
    make_intent,
    synthesize,
    target_payload,
)
from olivia_prompt.llm.slotfill import INSTRUCTION  # noqa: E402
from training.scripts.eval import check_expect, norm, score_case  # noqa: E402
from training.scripts.export_gguf import read_yaml_simple, write_merge_config  # noqa: E402


def rec(seed=0, i=0):
    return sample_batch(3, seed=seed)[i]


class TestIntents(unittest.TestCase):
    def test_short_is_short(self):
        for i in range(20):
            r = sample_batch(20, seed=1)[i]
            intent = make_intent(r, "short", __import__("random").Random(f"t:{i}"))
            self.assertLessEqual(len(intent), 24, intent)
            self.assertGreaterEqual(len(intent), 8, intent)

    def test_medium_mentions_light_and_camera(self):
        r = rec()
        intent = make_intent(r, "medium", __import__("random").Random("x"))
        self.assertIn("光", intent)
        self.assertTrue(intent.rstrip().endswith("。"))

    def test_long_carries_constraints(self):
        r = rec()
        intent = make_intent(r, "long", __import__("random").Random("y"))
        self.assertTrue(("清楚" in intent) or ("清晰" in intent), intent)
        self.assertIn("水印", intent)
        self.assertGreaterEqual(len(intent), 80)

    def test_pose_verb_embedded(self):
        from training.data.synthesize import POSE_VERB_ZH
        for r in sample_batch(10, seed=9):
            intent = make_intent(r, "short", __import__("random").Random(f"z:{r.id}"))
            verb = POSE_VERB_ZH[r.meta["pose"]]
            self.assertIn(verb, intent, f"{r.id}: {intent}")

    def test_multi_subject_prefix(self):
        r = rec()
        r.slots["subject_count"] = 2
        r.meta["subject_count"] = 2
        intent = make_intent(r, "short", __import__("random").Random("m"))
        self.assertIn("两个", intent)


class TestTargets(unittest.TestCase):
    def test_user_vs_inferred_split(self):
        r = rec()
        payload = target_payload(r, "long")
        for name in USER_SLOTS["long"]:
            self.assertEqual(payload[name]["src"], "user", name)
        self.assertEqual(payload["perch"]["src"], "inferred")
        self.assertEqual(payload["quality"]["src"], "default")

    def test_tier_visibility_in_target(self):
        r = rec()
        short_t = target_payload(r, "short")
        long_t = target_payload(r, "long")
        self.assertNotIn("handheld", short_t)
        self.assertIn("handheld", long_t)
        self.assertNotIn("framing", long_t)  # medium-only

    def test_pair_output_is_valid_slot_payload(self):
        r = rec()
        pair = build_pair(r, "long", __import__("random").Random("p"))
        self.assertIn("[档位:long]", pair["input"])
        target = json.loads(pair["output"])
        ss = SlotSet.from_mapping(target, tier="long")  # must parse cleanly
        self.assertFalse(has_errors([i for i in __import__("olivia_prompt.validate", fromlist=["validate"]).validate(ss)]))


class TestSynthesize(unittest.TestCase):
    def test_small_run_writes_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            stats = synthesize(20, seed=3, out_dir=tmp, hard_ratio=0.08)
            self.assertEqual(stats["pairs"], 60)  # 20 bases x 3 tiers
            self.assertEqual(stats["hard_multi_subject"], int(20 * 0.08))
            for name in ("train.jsonl", "val.jsonl", "dataset_info.json", "stats.json", "coverage.txt"):
                self.assertTrue((Path(tmp) / name).exists(), name)
            rows = [json.loads(l) for l in (Path(tmp) / "train.jsonl").read_text(encoding="utf-8").splitlines()]
            # serving contract: system role carries INSTRUCTION; user turn has no prompt prefix
            self.assertTrue(all({"system", "instruction", "input", "output"} == set(r) for r in rows))
            self.assertTrue(all(r["system"] == INSTRUCTION for r in rows))
            self.assertTrue(all(r["instruction"] == "" for r in rows))

    def test_dedup_cap_enforced(self):
        # 60 scenes * ~15 activities * 3 uses — 2000 bases must fit, 10000 must cap
        stats = synthesize(40, seed=5, out_dir=tempfile.mkdtemp())
        self.assertEqual(stats["bases"], 40)


class TestBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = [json.loads(l)
                     for l in (ROOT / "training" / "eval" / "benchmark.jsonl")
                     .read_text(encoding="utf-8").splitlines() if l.strip()]

    def test_50_cases_with_tier_mix(self):
        self.assertEqual(len(self.cases), 50)
        tiers = [c["tier"] for c in self.cases]
        self.assertEqual(tiers.count("short"), 18)
        self.assertEqual(tiers.count("medium"), 17)
        self.assertEqual(tiers.count("long"), 15)
        self.assertEqual(sum(1 for c in self.cases if c.get("hard")), 5)

    def test_expects_are_checkable(self):
        for case in self.cases:
            self.assertTrue(case["intent"].strip())
            self.assertTrue(case["expect"], case["id"])
            for e in case["expect"]:
                self.assertTrue(e["slot"])
                self.assertTrue("exact" in e or "contains" in e)


class TestEvalScoring(unittest.TestCase):
    def test_norm_strips_articles(self):
        self.assertEqual(norm("A quiet garden"), norm("quiet garden"))

    def test_check_expect_exact_and_contains(self):
        slots = SlotSet.from_mapping({"pose_family": "sit", "setting": "old study"}, tier="long")
        self.assertTrue(check_expect({"slot": "pose_family", "exact": "Sit"}, slots))
        self.assertTrue(check_expect({"slot": "setting", "contains": ["study"]}, slots))
        self.assertFalse(check_expect({"slot": "setting", "contains": ["beach"]}, slots))

    def test_contains_needs_majority_keywords(self):
        slots = SlotSet.from_mapping({"setting": "rooftop"}, tier="long")
        # 3 keywords, only 1 present -> fail (needs 2)
        self.assertFalse(check_expect(
            {"slot": "setting", "contains": ["rooftop", "garden", "study"]}, slots))
        # 2 of 3 -> pass
        garden = SlotSet.from_mapping({"setting": "quiet rooftop garden"}, tier="long")
        self.assertTrue(check_expect(
            {"slot": "setting", "contains": ["rooftop", "study", "garden"]}, garden))

    def test_score_case_perfect(self):
        case = {
            "expect": [
                {"slot": "pose_family", "exact": "sit"},
                {"slot": "gender", "exact": "female"},
            ],
        }
        slots = SlotSet.from_mapping({"pose_family": "sit", "gender": "female"}, tier="long")
        score = score_case(case, slots)
        self.assertEqual(score["expect_hits"], 2)
        self.assertEqual(score["weighted_hit"], score["weighted_sum"])
        self.assertEqual(score["errors"], [])

    def test_score_case_weighted_miss(self):
        case = {"expect": [
            {"slot": "pose_family", "exact": "sit"},   # high -> weight 3
            {"slot": "mood", "exact": "peaceful"},     # low -> weight 1
        ]}
        slots = SlotSet.from_mapping({"pose_family": "stand", "mood": "peaceful"}, tier="long")
        score = score_case(case, slots)
        self.assertEqual(score["weighted_hit"], 1)
        self.assertEqual(score["weighted_sum"], 4)


class TestExportHelpers(unittest.TestCase):
    def test_read_yaml_simple(self):
        cfg = read_yaml_simple(ROOT / "training" / "configs" / "lite-4b.yaml")
        self.assertEqual(cfg["lora_rank"], "32")
        self.assertEqual(cfg["learning_rate"], "8.0e-5")
        self.assertEqual(cfg["quantization_method"], "hqq")
        self.assertEqual(cfg["cutoff_len"], "1536")
        self.assertEqual(cfg["dataset_dir"], "training/data/out")

    def test_write_merge_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = write_merge_config(
                ROOT / "training" / "configs" / "lite-4b.yaml", Path(tmp) / "merged")
            text = dst.read_text(encoding="utf-8")
            self.assertIn("adapter_name_or_path: training/out/lite-4b", text)
            self.assertIn("export_dir:", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
