"""Generate the 50-case benchmark for M4 eval.

Committed as training/eval/benchmark.jsonl so eval runs are comparable across
checkpoints.  Each case: a Chinese intent (same synthesis path as training,
but a fixed held-out seed) + expected slot checks.  Enum-like slots are exact
matches after normalization; free-form slots (setting/action) use keyword
containment — a model may phrase "old study" differently without being wrong.

Regenerate only deliberately:
    python training/scripts/make_benchmark.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from olivia_prompt.sample import sample_batch  # noqa: E402
from training.data.synthesize import make_intent, USER_SLOTS  # noqa: E402

OUT = ROOT / "training" / "eval" / "benchmark.jsonl"

#: slot -> check style
EXACT = {
    "subject_count", "gender", "pose_family", "light_dir", "cam_height",
    "aspect", "medium", "hair_color", "outfit_main", "light_time",
}
CONTAINS = {"setting", "action"}

#: tier mix across the 50 cases
TIER_PLAN = (["short"] * 18) + (["medium"] * 17) + (["long"] * 15)

HARD_COUNT = 5
SEED = 7777  # held out from the training synthesis seed (42)


def _keywords(value: str) -> list:
    stop = {"a", "an", "the", "in", "on", "at", "with", "of", "her", "his"}
    return [w for w in value.lower().split() if w not in stop][:3]


def main() -> int:
    rng = random.Random(f"bench:{SEED}")
    records = sample_batch(50, seed=SEED)

    # 难例：双主体
    for rec in records[:HARD_COUNT]:
        rec.slots["subject_count"] = 2
        rec.meta["subject_count"] = 2

    rng.shuffle(records)
    cases = []
    for i, (rec, tier) in enumerate(zip(records, TIER_PLAN)):
        intent = make_intent(rec, tier, random.Random(f"bench-intent:{SEED}:{i}"))
        expect = []
        for name in USER_SLOTS[tier]:
            value = rec.slots.get(name)
            if value in (None, "", [], ()):
                continue
            value = str(value) if not isinstance(value, list) else value
            if name in EXACT:
                expect.append({"slot": name, "exact": value})
            elif name in CONTAINS:
                expect.append({"slot": name, "contains": _keywords(str(value))})
        cases.append({
            "id": f"bench-{i:03d}-{rec.id}-{tier}",
            "tier": tier,
            "intent": intent,
            "expect": expect,
            "hard": bool(i < HARD_COUNT),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"wrote {len(cases)} cases -> {OUT}")
    print(f"tiers: short={TIER_PLAN.count('short')} medium={TIER_PLAN.count('medium')} "
          f"long={TIER_PLAN.count('long')} hard={HARD_COUNT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
