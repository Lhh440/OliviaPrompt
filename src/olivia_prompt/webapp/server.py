"""HTTP server for the OliviaPrompt web UI.

Endpoints
---------
GET  /                 single page app
GET  /static/*         app.js / app.css
GET  /api/slots        slot registry (name, impact, kind, options, default, doc)
GET  /api/eco          target ecosystems
GET  /api/skeletons    long-tier skeleton variants
GET  /api/config       whether an LLM endpoint is configured (never returns the key)
POST /api/render       {slots, eco} -> three tiers + negative + issues
POST /api/ask          {intent, tier, eco, overrides?} -> slots + prompts (M2 pipeline)

Errors are JSON: {"error": "..."} with 4xx/5xx status.  Slot payload errors are
400s with the SlotError text — the UI shows them next to the table.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..ecosystem import ECOSYSTEMS, DEFAULT_ECOSYSTEM, get_ecosystem
from ..llm.client import resolve_config
from ..llm.pipeline import run_intent
from ..negative import build_negative
from ..render import render_all
from ..skeletons import SKELETON_DOCS, SKELETONS
from ..slots import IMPACT_ORDER, SLOTS, SLOT_ORDER, SlotError, SlotSet
from ..validate import has_errors, validate

_STATIC = Path(__file__).resolve().parent / "static"
_MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}


def _slot_rows() -> list:
    rows = []
    for name in SLOT_ORDER:
        s = SLOTS[name]
        default = list(s.default) if isinstance(s.default, tuple) else s.default
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
    return rows


def _slots_from_payload(payload: Any, tier: str = "long") -> SlotSet:
    if not isinstance(payload, dict):
        raise SlotError("槽位必须是 JSON 对象")
    return SlotSet.from_mapping(payload, tier=tier)


class Handler(BaseHTTPRequestHandler):
    server_version = "OliviaPrompt/0.2"

    # -- plumbing -----------------------------------------------------------
    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: Any) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def _body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"请求体不是合法 JSON：{exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return data

    def log_message(self, fmt: str, *args):  # keep the console clean
        pass

    # -- routing ------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._static_file("index.html")
        elif path.startswith("/static/"):
            self._static_file(path[len("/static/"):])
        elif path == "/api/slots":
            self._json(200, _slot_rows())
        elif path == "/api/eco":
            self._json(200, [{
                "key": e.key, "label": e.label, "strength": e.strength.value,
                "tag_style": e.tag_style, "cfg_hint": e.cfg_hint, "note": e.note,
            } for e in ECOSYSTEMS.values()])
        elif path == "/api/skeletons":
            self._json(200, [{"key": k, "doc": SKELETON_DOCS[k]} for k in SKELETONS])
        elif path == "/api/config":
            cfg = resolve_config()
            self._json(200, {
                "llm_ready": cfg is not None,
                "base_url": cfg.base_url if cfg else None,
                "model": cfg.model if cfg else None,
            })
        else:
            self._error(404, f"未知路径 {path}")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            body = self._body()
            if path == "/api/render":
                self._api_render(body)
            elif path == "/api/ask":
                self._api_ask(body)
            else:
                self._error(404, f"未知路径 {path}")
        except SlotError as exc:
            self._error(400, f"槽位错误：{exc}")
        except ValueError as exc:
            self._error(400, str(exc))
        except KeyError as exc:
            self._error(400, f"缺少字段：{exc}")
        except RuntimeError as exc:
            # e.g. LLM endpoint not configured
            self._error(503, str(exc))
        except Exception as exc:  # noqa: BLE001 — last-resort guard for the UI
            self._error(500, f"服务器内部错误：{exc}")

    # -- static -------------------------------------------------------------
    def _static_file(self, rel: str) -> None:
        if "/" in rel or ".." in rel or rel.startswith("."):
            self._error(404, "bad path")
            return
        file = _STATIC / rel
        if not file.is_file():
            self._error(404, f"no such file {rel}")
            return
        body = file.read_bytes()
        self._send(200, body, _MIME.get(file.suffix, "application/octet-stream"))

    # -- api ----------------------------------------------------------------
    def _api_render(self, body: Dict[str, Any]) -> None:
        eco = get_ecosystem(body.get("eco") or DEFAULT_ECOSYSTEM)
        skeleton = str(body.get("skeleton") or "A")
        slots = _slots_from_payload(body.get("slots") or {}, tier="long")
        issues = validate(slots)
        texts = render_all(slots, eco, skeleton=skeleton)
        ne = build_negative(slots, eco)
        self._json(200, {
            "ok": True,
            "ecosystem": {"key": eco.key, "label": eco.label, "strength": eco.strength.value},
            "slots": slots.as_dict(),
            "issues": [
                {"code": i.code, "severity": i.severity, "slot": i.slot, "message": i.message}
                for i in issues
            ],
            "prompts": {
                t: {
                    "positive": text,
                    "negative": ne.text,
                    "negative_strength": ne.strength,
                    "negative_note": ne.note,
                    "positive_patch": list(ne.positive_patch),
                }
                for t, text in texts.items()
            },
        })

    def _api_ask(self, body: Dict[str, Any]) -> None:
        intent = str(body.get("intent") or "").strip()
        if not intent:
            raise ValueError("intent 不能为空")
        tier = body.get("tier") or "long"
        result = run_intent(
            intent,
            tier=tier,
            ecosystem=body.get("eco") or DEFAULT_ECOSYSTEM,
            config=resolve_config(
                body.get("base_url"), body.get("api_key"), body.get("model")
            ),
            max_rounds=int(body.get("max_rounds") or 2),
            skeleton=str(body.get("skeleton") or "A"),
            compact=bool(body.get("compact")),
        )
        self._json(200, {
            "ok": True,
            "intent": result.intent,
            "tier": tier,
            "rounds": result.rounds,
            "slots": result.slots.as_dict(),
            "issues": [
                {"code": i.code, "severity": i.severity, "slot": i.slot, "message": i.message}
                for i in result.issues
            ],
            "prompts": result.prompts,
            "notes": result.notes,
            "has_errors": has_errors(result.issues),
        })


def make_server(host: str = "127.0.0.1", port: int = 7860) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)


def serve(host: str = "127.0.0.1", port: int = 7860, open_browser: bool = False) -> None:
    server = make_server(host, port)
    url = f"http://{host}:{port}"
    print(f"OliviaPrompt Web  ->  {url}  (Ctrl+C 退出)")
    if open_browser:
        import webbrowser
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        server.server_close()


def parse_bind(spec: Optional[str]) -> Tuple[str, int]:
    """'7860' | '127.0.0.1:7860' | '0.0.0.0:7860' -> (host, port)."""
    if not spec:
        return "127.0.0.1", 7860
    if ":" in spec:
        host, _, port = spec.partition(":")
        return host or "127.0.0.1", int(port)
    return "127.0.0.1", int(spec)


__all__ = ["Handler", "make_server", "serve", "parse_bind"]
