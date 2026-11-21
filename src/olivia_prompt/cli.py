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
            texts = render_all(ss, eco)
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
                    "positive": render(ss, args.tier, eco),
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
        _emit(render(ss, tier, eco))
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
            "  olivia-prompt render --slots examples/slots_rooftop.json --all\n"
            "  olivia-prompt init --tier medium > my.json\n"
            "  olivia-prompt slots\n"
            "  olivia-prompt eco\n"
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
