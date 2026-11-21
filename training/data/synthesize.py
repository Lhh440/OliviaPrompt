"""M4 — SFT data synthesizer.

Turns the M1 sampler's decided worlds into supervised pairs:

    input   = 中文画面需求（三种口吻：短 ≤15 字 / 中 ~60 字 / 长 ~150 字）
    output  = 该档位的完整槽位 JSON（{"slot": {"value", "src"}}）

The output is slot JSON, never prose — the renderers own every sentence, which
is what lets a 4B model run in the product path at all.

Provenance honesty: whatever the intent *actually says* is marked src=user in
the target; the rest is inferred; untouched defaults stay default.  That is
what teaches the model the user/inferred/default distinction instead of
hallucinating confidence floats.

Usage:
    python training/data/synthesize.py --n 2000 --seed 42 --out training/data/out
    python training/data/synthesize.py --n 200 --tiers short,medium

Dedup & gates (from the plan):
    * same (scene, activity) pair <= 3 uses            — enforced by the sampler
    * coverage matrix: open slots < 15 distinct values — reported, so you know
      what to widen before training rather than after
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from olivia_prompt.llm.slotfill import INSTRUCTION  # noqa: E402  (single source of truth)
from olivia_prompt.pools import LIGHT_DIR_ZH  # noqa: E402
from olivia_prompt.sample import SampleRecord, sample_batch  # noqa: E402
from olivia_prompt.slots import SLOTS, SLOT_ORDER  # noqa: E402

#: pose -> 在…… 之前的动词
POSE_VERB_ZH = {
    "sit": "坐在", "stand": "站在", "walk": "走在",
    "lean": "靠在", "kneel": "跪在", "lie": "躺在", "crouch": "蹲在",
}

#: slots the intent actually verbalizes, per tier -> src=user in the target
USER_SLOTS: Dict[str, frozenset] = {
    "short": frozenset({"subject_count", "gender", "hair_color", "setting", "action", "pose_family"}),
    "medium": frozenset({
        "subject_count", "gender", "hair_color", "setting", "action", "pose_family",
        "expression", "light_time", "light_dir", "cam_height", "aspect",
    }),
    "long": frozenset({
        "subject_count", "gender", "hair_color", "setting", "action", "pose_family",
        "expression", "light_time", "light_dir", "cam_height", "aspect",
        "outfit_main", "visibility",
    }),
}

#: renderer-injected, the model should always echo the standard default
ALWAYS_DEFAULT_SLOTS = frozenset({"quality"})


# --------------------------------------------------------------------------- #
# Chinese intent synthesis
# --------------------------------------------------------------------------- #

def _persona_zh(rec: SampleRecord) -> str:
    m = rec.meta
    hair = m["persona"]["hair_color_zh"]
    age = m["persona"]["age_zh"]
    noun = "女子" if m["gender"] == "female" else ("男子" if m["gender"] == "male" else "人")
    count = "" if str(m["subject_count"]) == "1" else "两个"
    return f"{count}{age}{hair}{noun}"


def _pose_verb(rec: SampleRecord) -> str:
    return POSE_VERB_ZH[str(rec.meta["pose"])]


def make_intent(rec: SampleRecord, tier: str, rng: random.Random) -> str:
    m = rec.meta
    p = _persona_zh(rec)
    place = m.get("name_zh") or ""
    zh_place = m.get("zh_place") or place
    act = m["activity_zh"]
    pose_v = _pose_verb(rec)
    expr = m["persona"]["expression_zh"]
    light = m["light_zh"]
    light_dir = LIGHT_DIR_ZH.get(m["light_dir"], m["light_dir"])
    cam = m["cam_height_zh"]
    aspect = m["aspect_zh"]
    outfit = m["wardrobe"]["outfit_main_zh"]
    detail = m["wardrobe"]["outfit_detail_zh"]

    if tier == "short":
        shape = rng.randrange(3)
        if shape == 0:
            return f"{p}{pose_v}{zh_place}{act}"
        if shape == 1:
            return f"{zh_place}{pose_v}{act}的{p}"
        return f"{pose_v}{zh_place}{act}的{p}"

    core = f"{p}{pose_v}{zh_place}{act}"
    if tier == "medium":
        shape = rng.randrange(2)
        if shape == 0:
            return f"{core}。{expr}，{light}的光线从{light_dir}照进来，{cam}，{aspect}。"
        return f"{core}，{expr}。{light}，光从{light_dir}来。{cam}，{aspect}。"

    # long: 叙事 + 约束
    shape = rng.randrange(2)
    body = (
        f"{p}，穿着{outfit}，衣服上有{detail}，{pose_v}{zh_place}{act}。"
        f"{expr}，{light}。镜头是{cam}，{aspect}。"
    )
    if shape == 0:
        tail = (
            "要求脸部和双手清晰可见，背景保持简洁不要抢走焦点，"
            "画面里不要出现任何文字或水印。"
        )
    else:
        tail = (
            "脸和手必须清楚完整，不要被裁切；背景收干净、层次分明；"
            "画面中不出现文字、水印或签名。"
        )
    return body + tail


# --------------------------------------------------------------------------- #
# target payload
# --------------------------------------------------------------------------- #

def _plain(value) -> object:
    if isinstance(value, tuple):
        return list(value)
    return value


def target_payload(rec: SampleRecord, tier: str) -> Dict[str, dict]:
    payload: Dict[str, dict] = {}
    user_slots = USER_SLOTS[tier]
    for name in SLOT_ORDER:
        slot = SLOTS[name]
        if not slot.in_tier(tier):
            continue
        raw = rec.slots.get(name)
        value = _plain(raw)
        if name in ALWAYS_DEFAULT_SLOTS or raw in (None, "", [], ()):
            src = "default"
        elif name in user_slots:
            src = "user"
        else:
            src = "inferred"
        payload[name] = {"value": value, "src": src}
    return payload


def build_pair(rec: SampleRecord, tier: str, rng: random.Random) -> dict:
    """Training row in the exact serving contract:

        system  = INSTRUCTION            (role-separated, never merged into the
                                          user turn — LF alpaca merges prompt+query,
                                          so `instruction` stays empty)
        user    = "[档位:t] intent"
        output  = full slot JSON
    """
    intent = make_intent(rec, tier, rng)
    return {
        "system": INSTRUCTION,
        "instruction": "",
        "input": f"[档位:{tier}] {intent}",
        "output": json.dumps(target_payload(rec, tier), ensure_ascii=False,
                             separators=(",", ":")),
        "meta": {"id": rec.id, "tier": tier, "scene": rec.meta["scene"],
                 "group": rec.meta["group"]},
    }


def rewrite_intent_with_llm(pairs: List[dict], client, max_items: Optional[int] = None) -> int:
    """Optional pass: a strong model rewrites the templated intents into
    natural phrasing.  Facts must survive; on any failure the template stays.
    Returns the number of rewritten intents."""
    import re

    from olivia_prompt.llm.slotfill import extract_json

    rewritten = 0
    for pair in (pairs[:max_items] if max_items else pairs):
        meta = pair["meta"]
        original = pair["input"].split("] ", 1)[-1]
        try:
            reply = client.chat([
                {"role": "system", "content":
                    "你是中文需求改写器。把给出的绘画需求改写成同一含义、更自然的口语表述。"
                    "保留全部事实（人物、场景、动作、光线、机位、约束），不要新增画面信息。"
                    '只输出 JSON：{"intent": "..."}'},
                {"role": "user", "content": original},
            ], temperature=0.9, max_tokens=300)
            new = str(extract_json(reply).get("intent") or "").strip()
            if new and 6 < len(new) < 400:
                tier = meta["tier"]
                pair["input"] = f"[档位:{tier}] {new}"
                rewritten += 1
        except Exception:
            continue
    return rewritten


# --------------------------------------------------------------------------- #
# main synthesis
# --------------------------------------------------------------------------- #

def synthesize(
    n: int,
    seed: int = 42,
    tiers: List[str] = ("short", "medium", "long"),
    hard_ratio: float = 0.0,
    out_dir: str = "training/data/out",
    val_ratio: float = 0.02,
    llm_client=None,
) -> dict:
    rng = random.Random(f"synth:{seed}")
    hard_n = int(n * hard_ratio)

    records = sample_batch(n, seed=seed)
    if len(records) < n:
        print(f"[warn] 采样器容量上限：只能产出 {len(records)}/{n} 条基准记录"
              "（场景×动作组合去重限制）——扩充活动池可解除。")

    # 难例：把前 hard_n 条改造成双主体
    for rec in records[:hard_n]:
        rec.slots["subject_count"] = 2
        rec.slots["gender"] = rec.meta["gender"]
        rec.meta["subject_count"] = 2
        rec.meta["hard"] = "multi_subject"

    pairs: List[dict] = []
    for rec in records:
        for tier in tiers:
            pairs.append(build_pair(rec, tier, random.Random(f"{rec.id}:{tier}")))

    if llm_client is not None:
        count = rewrite_intent_with_llm(pairs, llm_client)
        print(f"[llm] 改写意图 {count}/{len(pairs)} 条")

    rng.shuffle(pairs)
    val_count = max(1, int(len(pairs) * val_ratio)) if len(pairs) > 50 else 0
    val, train = pairs[:val_count], pairs[val_count:]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        with (out / name).open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(
                    {k: row[k] for k in ("system", "instruction", "input", "output")},
                    ensure_ascii=False) + "\n")

    # LLaMA-Factory dataset registration.
    # "system" column -> real system role at training time, so the fine-tuned
    # model sees exactly the same message shape the product/eval sends.
    # (alpaca merges prompt+query into the user turn; `instruction` is empty on
    #  purpose so the user turn is just "[档位:t] intent".)
    with (out / "dataset_info.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "olivia_prompt_sft": {
                "file_name": "train.jsonl",
                "columns": {"system": "system", "prompt": "instruction",
                            "query": "input", "response": "output"},
            }
        }, fh, ensure_ascii=False, indent=2)

    # stats + coverage gate
    from olivia_prompt.sample import coverage_report
    rep, flagged = coverage_report(records)
    stats = {
        "bases": len(records),
        "pairs": len(pairs),
        "train": len(train),
        "val": len(val),
        "hard_multi_subject": hard_n,
        "tiers": list(tiers),
        "seed": seed,
        "coverage_flagged": flagged,
    }
    with (out / "stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    with (out / "coverage.txt").open("w", encoding="utf-8") as fh:
        fh.write(rep + "\n")
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="M4 SFT 数据合成器")
    ap.add_argument("--n", type=int, default=2000, help="基准记录数（每条 × 档位数 = 总样本）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tiers", default="short,medium,long")
    ap.add_argument("--hard", type=float, default=0.08, help="双主体难例比例")
    ap.add_argument("--out", default=str(ROOT / "training" / "data" / "out"))
    ap.add_argument("--base-url", default=None, help="可选：用强模型改写意图的 OpenAI 端点")
    ap.add_argument("--model", default=None)
    args = ap.parse_args(argv)

    llm_client = None
    if args.base_url or args.model:
        from olivia_prompt.llm.client import resolve_config
        cfg = resolve_config(args.base_url, None, args.model)
        if cfg is None:
            print("[warn] LLM 未配置完整（需要 base_url 与 model），跳过改写")
        else:
            from olivia_prompt.llm.client import ChatClient
            llm_client = ChatClient(cfg)

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    stats = synthesize(args.n, seed=args.seed, tiers=tiers,
                       hard_ratio=args.hard, out_dir=args.out, llm_client=llm_client)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("\n下一步：把 out/ 目录注册进 LLaMA-Factory（dataset_info.json 已生成），"
          "然后运行 training/scripts/train.bat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
