"""Command line interface.

    olivia-prompt render --slots examples/slots_rooftop.json --tier long
    olivia-prompt render --slots examples/slots_rooftop.json --all --eco illustrious
    olivia-prompt render --json examples/slots_rooftop.json --json-out
    olivia-prompt init > my_slots.json
    olivia-prompt slots
    olivia-prompt check examples/slots_rooftop.json

M0 scope: the slot table comes from the user (or from ``init``).  Turning free
Chinese text into that table is M2, and the local model is M4 -- neither is
required to render, which is the whole point of splitting the work this way.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ecosystem import ECOSYSTEMS, ECOSYSTEM_CHOICES, DEFAULT_ECOSYSTEM, get_ecosystem
from .negative import build_negative
from .render import render, render_all
from .sample import MAX_SCENE_ACTIVITY_USES, coverage_matrix, coverage_report, sample_batch
from .scenes import SCENE_GROUPS, scene_brief, scenes_for_group
from .skeletons import SKELETONS
from .slots import (
    IMPACT_ORDER,
    SLOTS,
    SLOT_ORDER,
    SRC_LABELS,
    TIERS,
    SlotError,
    SlotSet,
)
from .validate import has_errors, validate


# --------------------------------------------------------------------------- #
# io helpers
# --------------------------------------------------------------------------- #

def _load(path: Optional[str]) -> Dict[str, Any]:
    if not path or path == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"JSON 解析失败: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit("槽位文件顶层必须是 JSON 对象")
    # allow {"slots": {...}} wrapping
    if "slots" in data and isinstance(data["slots"], dict):
        data = data["slots"]
    return data


def _emit(text: str) -> None:
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #

def cmd_render(args: argparse.Namespace) -> int:
    try:
        ss = SlotSet.from_mapping(_load(args.slots), tier=args.tier)
    except SlotError as exc:
        raise SystemExit(f"槽位错误: {exc}") from exc

    eco = get_ecosystem(args.eco)
    issues = validate(ss)

    if args.json_out:
        payload: Dict[str, Any] = {
            "tier": args.tier,
            "skeleton": args.skeleton,
            "ecosystem": {"key": eco.key, "label": eco.label,
                          "negative_strength": eco.strength.value},
            "slots": ss.as_dict(),
            "issues": [
                {"code": i.code, "severity": i.severity, "slot": i.slot,
                 "message": i.message}
                for i in issues
            ],
        }
        if args.all:
            texts = render_all(ss, eco, skeleton=args.skeleton)
            ne = build_negative(ss, eco)
            payload["prompts"] = {
                t: {"positive": txt, "negative": ne.text,
                    "negative_note": ne.note, "positive_patch": ne.positive_patch}
                for t, txt in texts.items()
            }
        else:
            ne = build_negative(ss, eco)
            payload["prompts"] = {
                args.tier: {
                    "positive": render(ss, args.tier, eco, skeleton=args.skeleton),
                    "negative": ne.text,
                    "negative_note": ne.note,
                    "positive_patch": ne.positive_patch,
                }
            }
        _emit(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if not has_errors(issues) else 2

    tiers: List[str] = list(TIERS) if args.all else [args.tier]
    ne = build_negative(ss, eco)

    for tier in tiers:
        _emit(f"=== POSITIVE · {tier} · {eco.label} ===")
        _emit(render(ss, tier, eco, skeleton=args.skeleton))
        _emit("")

    _emit(f"=== NEGATIVE · {eco.label} · strength={eco.strength.value} ===")
    _emit(ne.text if ne.text else "（本题材留空）")
    if ne.note:
        _emit(f"# {ne.note}")
    if ne.positive_patch:
        _emit("=== POSITIVE PATCH（负向无效，追加到正向末尾）===")
        _emit(", ".join(ne.positive_patch))
    _emit("")

    if issues:
        _emit("=== SELF-CHECK ===")
        for issue in issues:
            _emit(str(issue))

    return 0 if not has_errors(issues) else 2


def cmd_slots(args: argparse.Namespace) -> int:
    rows = []
    for name in SLOT_ORDER:
        s = SLOTS[name]
        default = s.default
        if isinstance(default, tuple):
            default = list(default)
        rows.append({
            "name": name,
            "impact": s.impact,
            "tiers": list(s.tiers),
            "kind": s.kind,
            "options": list(s.options),
            "default": default,
            "doc": s.doc,
        })
    rows.sort(key=lambda r: (IMPACT_ORDER[r["impact"]], r["name"]))
    if args.json_out:
        _emit(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0

    _emit(f"{'slot':<16} {'impact':<5} {'tiers':<18} default")
    _emit("-" * 78)
    for r in rows:
        tiers = ",".join(t[0] for t in r["tiers"])
        _emit(f"{r['name']:<16} {r['impact']:<5} {tiers:<18} {r['default']!r}")
        if r["options"]:
            _emit(f"{'':<16} options: {', '.join(r['options'])}")
    _emit("")
    _emit("tiers: s=short m=medium l=long   impact: high/mid/low")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    payload: Dict[str, Any] = {}
    for name in SLOT_ORDER:
        s = SLOTS[name]
        if not s.in_tier(args.tier):
            continue
        value: Any = list(s.default) if isinstance(s.default, tuple) else s.default
        payload[name] = {"value": value, "src": "default"}
    payload["_sources"] = {}
    _emit(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    try:
        ss = SlotSet.from_mapping(_load(args.slots), tier=args.tier)
    except SlotError as exc:
        raise SystemExit(f"槽位错误: {exc}") from exc
    issues = validate(ss)
    if not issues:
        _emit("OK — 没有发现问题。")
        return 0
    for issue in issues:
        _emit(str(issue))
    _emit("")
    _emit(f"共 {len(issues)} 项，其中 error {sum(1 for i in issues if i.severity == 'error')} 项。")
    return 0 if not has_errors(issues) else 2


def _sample_records(args: argparse.Namespace):
    return sample_batch(args.n, seed=args.seed, groups=args.groups or None)


def cmd_sample(args: argparse.Namespace) -> int:
    records = _sample_records(args)
    eco = get_ecosystem(args.eco)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = [r.as_row() for r in records]
    with (out / "samples.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.render:
        with (out / "prompts.jsonl").open("w", encoding="utf-8") as fh:
            for r in records:
                ss = SlotSet.from_mapping(r.slots, tier="long")
                texts = render_all(ss, eco, skeleton=args.skeleton)
                ne = build_negative(ss, eco)
                fh.write(json.dumps({
                    "id": r.id,
                    "positive": texts,
                    "negative": {
                        "text": ne.text, "strength": ne.strength,
                        "note": ne.note, "positive_patch": ne.positive_patch,
                    },
                }, ensure_ascii=False) + "\n")

    rep, flagged = coverage_report(records)
    with (out / "coverage.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "n": len(records),
            "seed": args.seed,
            "groups": args.groups or ["all"],
            "matrix": _matrix_json(records),
            "flagged": flagged,
        }, fh, ensure_ascii=False, indent=2)

    _emit(f"生成 {len(records)} 条采样记录 -> {out / 'samples.jsonl'}")
    if args.render:
        _emit(f"三档渲染 + 负向 -> {out / 'prompts.jsonl'}（生态 {eco.key}）")
    _emit(f"覆盖矩阵 -> {out / 'coverage.json'}")
    _emit("")
    _emit(rep)
    return 0


def cmd_coverage(args: argparse.Namespace) -> int:
    records = _sample_records(args)
    rep, flagged = coverage_report(records)
    if args.json_out:
        _emit(json.dumps(
            {"n": len(records), "matrix": _matrix_json(records), "flagged": flagged},
            ensure_ascii=False, indent=2,
        ))
        return 0
    _emit(f"n={len(records)} seed={args.seed}")
    _emit("")
    _emit(rep)
    return 0


def cmd_scenes(args: argparse.Namespace) -> int:
    scenes = scenes_for_group(",".join(args.groups) if args.groups else None)
    if args.json_out:
        _emit(json.dumps([scene_brief(s) for s in scenes], ensure_ascii=False, indent=2))
        return 0
    _emit(f"{'id':<22} {'group':<15} {'in':<3} poses    场景")
    _emit("-" * 84)
    for s in scenes:
        poses = ",".join(p[0] for p in s.fit_poses)
        _emit(f"{s.id:<22} {s.group:<15} {'y' if s.indoor else 'n':<3} {poses:<8} {s.name_zh} · {s.light_zh}")
    _emit("")
    _emit(f"共 {len(scenes)} 个场景；groups: {', '.join(SCENE_GROUPS)}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from .llm.client import resolve_config
    from .llm.pipeline import run_intent

    cfg = resolve_config(args.base_url, args.api_key, args.model)
    if cfg is None:
        _emit("未配置模型。三种方式任选其一：")
        _emit("  1. 环境变量  OLIVIA_BASE_URL / OLIVIA_MODEL （可选 OLIVIA_API_KEY）")
        _emit('  2. 配置文件  ~/.olivia-prompt/config.json  {"base_url": "...", "model": "..."}')
        _emit("  3. 命令行    --base-url http://127.0.0.1:8081/v1 --model qwen3-4b")
        _emit("本地 GGUF：llama-server -m models/xxx.gguf --port 8081 即是 OpenAI 兼容端点。")
        return 2

    try:
        result = run_intent(
            args.intent, tier=args.tier, ecosystem=args.eco,
            config=cfg, max_rounds=args.max_rounds, compact=args.compact,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    eco = get_ecosystem(args.eco)
    if args.json_out:
        payload = {
            "intent": result.intent,
            "tier": args.tier,
            "ecosystem": {"key": eco.key, "label": eco.label},
            "rounds": result.rounds,
            "slots": result.slots.as_dict(),
            "issues": [
                {"code": i.code, "severity": i.severity, "slot": i.slot, "message": i.message}
                for i in result.issues
            ],
            "prompts": result.prompts,
            "notes": result.notes,
        }
        _emit(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if not has_errors(result.issues) else 2

    _emit(f"=== 意图 ===\n{result.intent}")
    _emit(f"\n=== 模型 ===\n{cfg.describe()}（{result.rounds} 轮）")
    if result.notes:
        _emit("\n=== 备注 ===")
        for note in result.notes:
            _emit(f"# {note}")

    if args.show_slots:
        _emit("\n=== 槽位表（src: u=user i=inferred d=default）===")
        for name in SLOT_ORDER:
            s = SLOTS[name]
            if not s.in_tier(args.tier):
                continue
            mark = {"user": "u", "inferred": "i", "default": "d"}[result.slots.source(name)]
            value = result.slots[name]
            _emit(f"  [{mark}] {name:<16} {value!r}")

    for tier, block in result.prompts.items():
        _emit(f"\n=== POSITIVE · {tier} · {eco.label} ===")
        _emit(str(block["positive"]))

    neg = result.prompts[args.tier]
    _emit(f"\n=== NEGATIVE · {eco.label} · strength={neg['negative_strength']} ===")
    _emit(str(neg["negative"]) or "（本生态留空）")
    if neg["negative_note"]:
        _emit(f"# {neg['negative_note']}")
    if neg["positive_patch"]:
        _emit("=== POSITIVE PATCH（追加到正向末尾）===")
        _emit(", ".join(neg["positive_patch"]))

    if result.issues:
        _emit("\n=== SELF-CHECK ===")
        for issue in result.issues:
            _emit(str(issue))

    return 0 if not has_errors(result.issues) else 2


def _matrix_json(records) -> dict:
    matrix = coverage_matrix(records)
    return {slot: {str(val): cnt for val, cnt in counter.items()}
            for slot, counter in matrix.items()}


def cmd_web(args: argparse.Namespace) -> int:
    from .webapp import serve, parse_bind
    host, port = parse_bind(args.bind)
    serve(host, port, open_browser=args.open)
    return 0


def cmd_eco(args: argparse.Namespace) -> int:
    if args.json_out:
        rows = [
            {"key": e.key, "label": e.label, "negative_strength": e.strength.value,
             "tag_style": e.tag_style, "cfg_hint": e.cfg_hint, "note": e.note}
            for e in ECOSYSTEMS.values()
        ]
        _emit(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    _emit(f"{'key':<14} {'strength':<7} {'style':<7} label")
    _emit("-" * 74)
    for e in ECOSYSTEMS.values():
        mark = "*" if e.key == DEFAULT_ECOSYSTEM else " "
        _emit(f"{e.key:<14} {e.strength.value:<7} {e.tag_style:<7} {mark}{e.label}")
        _emit(f"{'':<14} {e.cfg_hint} — {e.note}")
    _emit("")
    _emit("* = 默认生态")
    return 0


def cmd_unresolved(args: argparse.Namespace) -> int:
    """Report high-impact slots the user never actually specified."""
    try:
        ss = SlotSet.from_mapping(_load(args.slots), tier=args.tier)
    except SlotError as exc:
        raise SystemExit(f"槽位错误: {exc}") from exc
    pending = ss.open_questions()
    if not pending:
        _emit("所有高影响槽位都有信息，不需要追问。")
        return 0
    _emit(f"以下 {len(pending)} 个高影响槽位仍无信息，是可追问的候选：")
    for s in pending:
        opts = f"  选项: {', '.join(s.options)}" if s.options else ""
        _emit(f"  - {s.name}（{SRC_LABELS[ss.source(s.name)]}）{s.doc}{opts}")
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="olivia-prompt",
        description="把槽位表渲染成三档正向提示词 + 五层镜像负向提示词。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  olivia-prompt render --slots examples/slots_rooftop.json --tier long\n"
            "  olivia-prompt render --slots examples/slots_rooftop.json --all --skeleton B\n"
            "  olivia-prompt init --tier medium > my.json\n"
            "  olivia-prompt slots\n"
            "  olivia-prompt eco\n"
            "  olivia-prompt scenes\n"
            "  olivia-prompt sample --n 200 --render --out samples\n"
            "  olivia-prompt coverage --n 200\n"
            "  olivia-prompt ask \"屋顶花园画画的白发女孩\" --show-slots\n"
            "  olivia-prompt web --open\n"
        ),
    )
    p.add_argument("-V", "--version", action="store_true", help="打印版本并退出")
    sub = p.add_subparsers(dest="command")

    def add_common(sp: argparse.ArgumentParser, require_slots: bool = True) -> None:
        if require_slots:
            sp.add_argument("slots", nargs="?", default="-",
                            help="槽位 JSON 文件，- 或省略则读 stdin")
        sp.add_argument("--tier", choices=TIERS, default="long", help="档位（默认 long）")
        sp.add_argument("--eco", choices=ECOSYSTEM_CHOICES, default=DEFAULT_ECOSYSTEM,
                        help=f"目标生态（默认 {DEFAULT_ECOSYSTEM}）")
        sp.add_argument("--json-out", action="store_true", help="输出结构化 JSON")

    r = sub.add_parser("render", help="渲染正向 + 负向提示词")
    add_common(r)
    r.add_argument("--all", action="store_true", help="一次输出三档")
    r.add_argument("--skeleton", choices=SKELETONS, default="A",
                   help="长档骨架变体（默认 A）")
    r.set_defaults(func=cmd_render)

    s = sub.add_parser("slots", help="列出全部槽位定义")
    add_common(s, require_slots=False)
    s.set_defaults(func=cmd_slots, tier="long", eco=DEFAULT_ECOSYSTEM)

    i = sub.add_parser("init", help="生成一份可编辑的槽位模板")
    add_common(i, require_slots=False)
    i.set_defaults(func=cmd_init, eco=DEFAULT_ECOSYSTEM)

    c = sub.add_parser("check", help="只跑自洽校验")
    add_common(c)
    c.set_defaults(func=cmd_check)

    e = sub.add_parser("eco", help="列出目标生态与负向有效性")
    add_common(e, require_slots=False)
    e.set_defaults(func=cmd_eco, tier="long")

    u = sub.add_parser("unresolved", help="列出无信息的高影响槽位（追问候选）")
    add_common(u)
    u.set_defaults(func=cmd_unresolved)

    smp = sub.add_parser("sample", help="规则版批量采样：场景库 × 槽位（M1）")
    smp.add_argument("--n", type=int, default=200, help="采样条数（默认 200）")
    smp.add_argument("--seed", type=int, default=0, help="随机种子（默认 0，可复现）")
    smp.add_argument("--groups", nargs="*", choices=SCENE_GROUPS,
                     help="场景组过滤，默认全部")
    smp.add_argument("--out", default="samples", help="输出目录（默认 ./samples）")
    smp.add_argument("--render", action="store_true", help="同时输出三档渲染 + 负向")
    smp.add_argument("--eco", choices=ECOSYSTEM_CHOICES, default=DEFAULT_ECOSYSTEM)
    smp.add_argument("--skeleton", choices=SKELETONS, default="A")
    smp.set_defaults(func=cmd_sample)

    cov = sub.add_parser("coverage", help="槽位覆盖矩阵（采样结果的预训练门禁）")
    cov.add_argument("--n", type=int, default=200)
    cov.add_argument("--seed", type=int, default=0)
    cov.add_argument("--groups", nargs="*", choices=SCENE_GROUPS)
    cov.add_argument("--json-out", action="store_true")
    cov.set_defaults(func=cmd_coverage)

    sc = sub.add_parser("scenes", help="列出场景库")
    sc.add_argument("--groups", nargs="*", choices=SCENE_GROUPS)
    sc.add_argument("--json-out", action="store_true")
    sc.set_defaults(func=cmd_scenes)

    a = sub.add_parser("ask", help="中文需求 → 槽位 → 完整正反向提示词（M2）")
    a.add_argument("intent", help="中文画面需求，例如「屋顶花园画画的白发女孩」")
    add_common(a, require_slots=False)
    a.add_argument("--base-url", default=None, help="OpenAI 兼容端点（默认读环境/配置文件）")
    a.add_argument("--api-key", default=None)
    a.add_argument("--model", default=None)
    a.add_argument("--max-rounds", type=int, default=2, help="自洽修复轮次上限")
    a.add_argument("--compact", action="store_true",
                   help="用微调模型的紧凑契约提示（system=INSTRUCTION，user=[档位] 需求）")
    a.add_argument("--show-slots", action="store_true", help="打印槽位表及来源")
    a.set_defaults(func=cmd_ask)

    w = sub.add_parser("web", help="启动 Web 端（槽位表可编辑 + 三档并列 + 负向有效性）")
    w.add_argument("--bind", default="127.0.0.1:7860",
                   help="监听地址，host:port 或纯端口（默认 127.0.0.1:7860）")
    w.add_argument("--open", action="store_true", help="启动后打开浏览器")
    w.set_defaults(func=cmd_web)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "version", False):
        from . import __version__
        print(f"olivia-prompt {__version__}")
        return 0
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
