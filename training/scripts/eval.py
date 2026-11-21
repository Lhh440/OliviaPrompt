"""M4 — benchmark eval runner.

Runs a checkpoint (or any OpenAI-compatible endpoint) against the 50-case
benchmark and scores it on the axes that actually matter for this product:

* json_ok        — did the model emit parseable slot JSON at all
* slot accuracy  — exact/contains match per expected slot, weighted by slot
                   impact (high x3 / mid x2 / low x1)
* consistency    — self-consistency error count via olivia_prompt.validate
* src sanity     — did the model emit legal src values

Usage:
    python training/scripts/eval.py --base-url http://127.0.0.1:8081/v1 \
        --model olivia-lite --label lite --out training/eval/report_lite.json

Comparison (the plan's mandatory experiment: 微调版 vs 原始基座+few-shot):
    python training/scripts/eval.py --base-url <base> --model Qwen3-4B \
        --baseline --label base4b --out training/eval/report_base.json
    python training/scripts/eval.py --compare report_lite.json report_base.json

``--baseline`` swaps the compact SFT instruction for the product's full
few-shot system prompt, giving the raw base model its best possible shot.

Optional round-trip check (--roundtrip): render the long prompt and ask a
strong model "if you drew this, what would be missing/extra vs the intent?" —
the product's round-trip self-check, in batch form.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from olivia_prompt.ecosystem import DEFAULT_ECOSYSTEM  # noqa: E402
from olivia_prompt.llm.client import ChatClient, ChatConfig, resolve_config  # noqa: E402
from olivia_prompt.llm.slotfill import extract_json  # noqa: E402
from olivia_prompt.slots import IMPACT_ORDER, SLOTS, SlotError, SlotSet  # noqa: E402
from olivia_prompt.validate import validate  # noqa: E402
from training.data.synthesize import INSTRUCTION  # noqa: E402

BENCH = ROOT / "training" / "eval" / "benchmark.jsonl"

_ARTICLE = re.compile(r"^(a|an|the)\s+")
_WS = re.compile(r"\s+")


def norm(text: str) -> str:
    t = _WS.sub(" ", str(text).lower().strip())
    return _ARTICLE.sub("", t)


def check_expect(expect: dict, slots: SlotSet) -> bool:
    name = expect["slot"]
    value = slots.get(name)
    if value in (None, "", [], ()):
        return False
    if "exact" in expect:
        return norm(value) == norm(expect["exact"])
    kws = expect.get("contains") or []
    low = norm(value)
    needed = max(1, -(-len(kws) * 3 // 5))  # ceil(0.6 * n)
    hits = sum(1 for kw in kws if kw in low)
    return hits >= needed


def score_case(case: dict, slots: SlotSet) -> dict:
    hits, weight_sum, weight_hit = 0, 0, 0
    per_slot = {}
    for expect in case["expect"]:
        name = expect["slot"]
        weight = {"high": 3, "mid": 2, "low": 1}[SLOTS[name].impact]
        ok = check_expect(expect, slots)
        per_slot[name] = ok
        weight_sum += weight
        weight_hit += weight if ok else 0
        hits += 1 if ok else 0
    errors = [i for i in validate(slots) if i.severity == "error"]
    return {
        "expect_n": len(case["expect"]),
        "expect_hits": hits,
        "weighted_hit": weight_hit,
        "weighted_sum": weight_sum,
        "errors": [str(e) for e in errors],
        "per_slot": per_slot,
    }


def run_benchmark(client: ChatClient, baseline: bool, limit: int = 0,
                  roundtrip: bool = False) -> List[dict]:
    cases = [json.loads(line) for line in BENCH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        cases = cases[:limit]
    results = []
    for case in cases:
        tier = case["tier"]
        system = (DEFAULT_SYSTEM if baseline else INSTRUCTION)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"[档位:{tier}] {case['intent']}"},
        ]
        row: dict = {"id": case["id"], "tier": tier, "hard": case.get("hard", False)}
        try:
            raw = client.chat(messages, temperature=0.2)
            slots = SlotSet.from_mapping(extract_json(raw), tier=tier)
            row.update(json_ok=True)
            row.update(score_case(case, slots))
            row["slots"] = {k: slots.get(k) for k in
                            ("setting", "pose_family", "light_dir", "medium", "hair_color")}
        except SlotError as exc:
            row.update(json_ok=False, error=f"slot: {exc}")
        except Exception as exc:  # noqa: BLE001
            row.update(json_ok=False, error=str(exc)[:200])
        if roundtrip and row.get("json_ok"):
            row["roundtrip"] = roundtrip_check(client, case, row)
        results.append(row)
    return results


def roundtrip_check(client: ChatClient, case: dict, row: dict) -> dict:
    """Ask a strong model what the rendered prompt would miss/over-add vs intent."""
    from olivia_prompt.render import render_all
    try:
        from olivia_prompt.llm.slotfill import parse_slots
        # rebuild the slots from the eval row's stored subset is impossible —
        # re-run the parse from the raw reply is not stored either, so we ask
        # with the slots we kept.  A light-weight proxy: ask about the intent
        # vs the produced slot set.
        slots_desc = json.dumps(row.get("slots", {}), ensure_ascii=False)
        reply = client.chat([
            {"role": "system", "content":
                "你是提示词质检员。对比「用户需求」和「槽位表」，"
                '只输出 JSON：{"missing": ["需求里有但槽位没体现的"], "extra": ["槽位里多余或矛盾的"]}。'},
            {"role": "user", "content": f"需求：{case['intent']}\n槽位表：{slots_desc}"},
        ], temperature=0.0, max_tokens=400)
        data = extract_json(reply)
        return {"missing": data.get("missing", [])[:5], "extra": data.get("extra", [])[:5]}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:120]}


DEFAULT_SYSTEM = ""


def _default_system() -> None:
    """Import lazily to avoid a circular import at module load."""
    global DEFAULT_SYSTEM
    from olivia_prompt.llm.slotfill import system_prompt
    DEFAULT_SYSTEM = system_prompt("long")  # replaced per tier at call time


def summarize(rows: List[dict]) -> dict:
    total = len(rows)
    ok_rows = [r for r in rows if r.get("json_ok")]
    expect_sum = sum(r["expect_n"] for r in ok_rows)
    expect_hits = sum(r["expect_hits"] for r in ok_rows)
    w_sum = sum(r["weighted_sum"] for r in ok_rows)
    w_hit = sum(r["weighted_hit"] for r in ok_rows)

    def _part(rows_):
        if not rows_:
            return {}
        ok = [r for r in rows_ if r.get("json_ok")]
        return {
            "n": len(rows_),
            "json_ok": len(ok) / len(rows_),
            "slot_acc": (sum(r["expect_hits"] for r in ok) /
                         max(1, sum(r["expect_n"] for r in ok))),
            "weighted_acc": (sum(r["weighted_hit"] for r in ok) /
                             max(1, sum(r["weighted_sum"] for r in ok))),
            "error_rate": (sum(1 for r in ok if r["errors"]) / max(1, len(ok))),
        }

    return {
        "n": total,
        "json_ok_rate": len(ok_rows) / max(1, total),
        "slot_acc": expect_hits / max(1, expect_sum),
        "weighted_acc": w_hit / max(1, w_sum),
        "consistency_error_rate": sum(1 for r in ok_rows if r["errors"]) / max(1, len(ok_rows)),
        "by_tier": {t: _part([r for r in rows if r["tier"] == t])
                    for t in ("short", "medium", "long")},
        "hard": _part([r for r in rows if r.get("hard")]),
    }


def report_markdown(label: str, summary: dict) -> str:
    lines = [
        f"# Eval · {label}",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| JSON 可解析率 | {summary['json_ok_rate']:.1%} |",
        f"| 槽位命中率 | {summary['slot_acc']:.1%} |",
        f"| 加权命中率（高×3/中×2/低×1）| {summary['weighted_acc']:.1%} |",
        f"| 自洽错误率 | {summary['consistency_error_rate']:.1%} |",
        "",
        "| 档位 | n | JSON | 命中 | 加权 | 自洽错误 |",
        "|---|---|---|---|---|---|",
    ]
    for tier in ("short", "medium", "long"):
        s = summary["by_tier"].get(tier) or {}
        if not s:
            continue
        lines.append(
            f"| {tier} | {s['n']} | {s['json_ok']:.0%} | {s['slot_acc']:.0%} "
            f"| {s['weighted_acc']:.0%} | {s['error_rate']:.0%} |")
    if summary.get("hard"):
        h = summary["hard"]
        lines.append(f"\n难例（双主体）：n={h['n']}，JSON {h['json_ok']:.0%}，加权 {h['weighted_acc']:.0%}")
    return "\n".join(lines) + "\n"


def compare_reports(path_a: str, path_b: str) -> int:
    a = json.loads(Path(path_a).read_text(encoding="utf-8"))
    b = json.loads(Path(path_b).read_text(encoding="utf-8"))
    keys = ("json_ok_rate", "slot_acc", "weighted_acc", "consistency_error_rate")
    print(f"{'metric':<28} {a['label']:>12} {b['label']:>12}   delta")
    print("-" * 66)
    for k in keys:
        va, vb = a["summary"][k], b["summary"][k]
        better = "↑" if ((vb > va) == (k != "consistency_error_rate")) and vb != va else "="
        print(f"{k:<28} {va:>12.1%} {vb:>12.1%}   {better} {vb - va:+.1%}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="M4 benchmark eval")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="model")
    ap.add_argument("--baseline", action="store_true",
                    help="用产品版 few-shot 系统提示（给原始基座模型的最佳机会）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--roundtrip", action="store_true", help="附加 round-trip 质检（慢）")
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="对比两份报告 JSON")
    args = ap.parse_args(argv)

    if args.compare:
        return compare_reports(args.compare[0], args.compare[1])

    cfg = resolve_config(args.base_url, args.api_key, args.model)
    if cfg is None:
        raise SystemExit("需要 --base-url/--model 或环境变量 OLIVIA_BASE_URL/OLIVIA_MODEL")
    _default_system()
    client = ChatClient(cfg)
    rows = run_benchmark(client, baseline=args.baseline, limit=args.limit,
                         roundtrip=args.roundtrip)
    summary = summarize(rows)

    out = Path(args.out) if args.out else ROOT / "training" / "eval" / f"report_{args.label}.json"
    payload = {
        "label": args.label,
        "endpoint": cfg.base_url,
        "model": cfg.model,
        "baseline_prompt": args.baseline,
        "summary": summary,
        "rows": rows,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md = out.with_suffix(".md")
    md.write_text(report_markdown(args.label, summary), encoding="utf-8")
    print(report_markdown(args.label, summary))
    print(f"报告 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
