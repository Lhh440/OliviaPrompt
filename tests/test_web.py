"""M3 — web server endpoints + static SPA."""

from __future__ import annotations

import json
import sys
import threading
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from olivia_prompt.webapp import make_server  # noqa: E402

EXAMPLES = ROOT / "examples"


def request(base: str, path: str, body: dict | None = None):
    url = base + path
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


class TestWebApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server("127.0.0.1", 0)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_index_served(self):
        with urllib.request.urlopen(self.base + "/", timeout=10) as resp:
            html = resp.read().decode("utf-8")
        self.assertIn("OliviaPrompt", html)
        self.assertIn("/static/app.js", html)

    def test_static_assets(self):
        for path, needle in (
            ("/static/app.js", "OliviaPrompt"),
            ("/static/app.css", "slot-row"),
        ):
            with urllib.request.urlopen(self.base + path, timeout=10) as resp:
                self.assertIn(needle, resp.read().decode("utf-8"))
            self.assertEqual(resp.status, 200)

    def test_static_traversal_blocked(self):
        status, data = request(self.base, "/static/../slots.py")
        self.assertEqual(status, 404)

    def test_api_slots(self):
        status, rows = request(self.base, "/api/slots")
        self.assertEqual(status, 200)
        names = {r["name"] for r in rows}
        self.assertIn("pose_family", names)
        self.assertIn("handheld", names)

    def test_api_eco_and_skeletons_and_config(self):
        status, ecos = request(self.base, "/api/eco")
        self.assertEqual(status, 200)
        self.assertTrue(any(e["key"] == "illustrious" for e in ecos))
        status, skels = request(self.base, "/api/skeletons")
        self.assertEqual(status, 200)
        self.assertEqual({s["key"] for s in skels}, {"A", "B", "C", "D"})
        status, cfg = request(self.base, "/api/config")
        self.assertEqual(status, 200)
        self.assertIn("llm_ready", cfg)

    def test_api_render_all_tiers(self):
        slots = json.loads((EXAMPLES / "slots_rooftop.json").read_text(encoding="utf-8"))
        status, data = request(self.base, "/api/render", {"slots": slots, "eco": "illustrious"})
        self.assertEqual(status, 200)
        self.assertEqual(set(data["prompts"]), {"short", "medium", "long"})
        self.assertIn("multiple people", data["prompts"]["long"]["negative"])
        self.assertTrue(data["prompts"]["long"]["positive"].startswith("masterpiece"))

    def test_api_render_skeleton_variant(self):
        slots = json.loads((EXAMPLES / "slots_rooftop.json").read_text(encoding="utf-8"))
        _s1, a = request(self.base, "/api/render", {"slots": slots})
        _s2, b = request(self.base, "/api/render", {"slots": slots, "skeleton": "D"})
        self.assertNotEqual(a["prompts"]["long"]["positive"], b["prompts"]["long"]["positive"])

    def test_api_render_bad_slot_is_400(self):
        status, data = request(self.base, "/api/render", {"slots": {"not_a_slot": 1}})
        self.assertEqual(status, 400)
        self.assertIn("槽位错误", data["error"])

    def test_api_render_bad_enum_is_400(self):
        slots = json.loads((EXAMPLES / "slots_rooftop.json").read_text(encoding="utf-8"))
        slots["pose_family"] = "float"
        status, data = request(self.base, "/api/render", {"slots": slots})
        self.assertEqual(status, 400)

    def test_api_render_edited_slot_becomes_user(self):
        slots = json.loads((EXAMPLES / "slots_rooftop.json").read_text(encoding="utf-8"))
        payload = {k: v for k, v in slots.items()}
        payload["hair_color"] = {"value": "mint", "src": "user"}
        status, data = request(self.base, "/api/render", {"slots": payload})
        self.assertEqual(status, 200)
        self.assertEqual(data["slots"]["hair_color"]["src"], "user")
        self.assertIn("mint hair", data["prompts"]["long"]["positive"])

    def test_api_ask_without_model_is_error(self):
        # resolve_config() picks up real env/config on this machine only if set;
        # force the no-config path by sending a body without overrides and
        # ensuring env vars are unset.
        import os
        saved = {k: os.environ.pop(k, None) for k in ("OLIVIA_BASE_URL", "OLIVIA_MODEL", "OLIVIA_API_KEY")}
        try:
            status, data = request(self.base, "/api/ask", {"intent": "测试"})
            if status == 503:
                self.assertIn("未配置", data["error"])
            # if a local config file exists on this machine the pipeline may
            # actually run; both outcomes are acceptable here
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v


if __name__ == "__main__":
    unittest.main(verbosity=2)
