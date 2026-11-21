"""M2 — LLM adapter: prompt building, JSON extraction, repair loop, pipeline."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from olivia_prompt.ecosystem import get_ecosystem  # noqa: E402
from olivia_prompt.llm.client import ChatClient, ChatConfig, ChatError, resolve_config  # noqa: E402
from olivia_prompt.llm.pipeline import run_intent  # noqa: E402
from olivia_prompt.llm.slotfill import extract_json, fill_slots, parse_slots, system_prompt  # noqa: E402
from olivia_prompt.slots import SlotError  # noqa: E402


def rooftop_payload() -> dict:
    raw = json.loads((ROOT / "examples" / "slots_rooftop.json").read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


class FakeClient:
    """Scripted client: pops canned replies, records every message list."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list = []

    def chat(self, messages, **kwargs):
        self.calls.append([dict(m) for m in messages])
        if not self.replies:
            raise AssertionError("FakeClient ran out of scripted replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _payload_json() -> str:
    return json.dumps(rooftop_payload(), ensure_ascii=False)


class TestExtractJson(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_fenced(self):
        text = "好的，这是槽位表：\n```json\n{\"a\": {\"value\": 1}}\n```\n以上。"
        self.assertEqual(extract_json(text), {"a": {"value": 1}})

    def test_prose_wrapped(self):
        text = '下面是结果 {"gender": {"value": "female", "src": "user"}} 请查收'
        self.assertEqual(extract_json(text)["gender"]["value"], "female")

    def test_trailing_comma_repaired(self):
        self.assertEqual(extract_json('{"a": 1, "b": 2,}'), {"a": 1, "b": 2})

    def test_nested_braces_in_strings(self):
        text = '{"note": {"value": "a } brace", "src": "inferred"}}'
        self.assertEqual(extract_json(text)["note"]["value"], "a } brace")

    def test_garbage_raises_slot_error(self):
        with self.assertRaises(SlotError):
            extract_json("我觉得应该是坐着的")
        with self.assertRaises(SlotError):
            extract_json('{"a": ')
        with self.assertRaises(SlotError):
            extract_json("")


class TestParseSlots(unittest.TestCase):
    def test_unknown_slot_rejected(self):
        bad = dict(rooftop_payload())
        bad["hair_colour"] = "silver"
        with self.assertRaises(SlotError):
            parse_slots(json.dumps(bad), tier="long")

    def test_null_values_fall_back_to_default(self):
        raw = rooftop_payload()
        raw["mood"] = None
        ss = parse_slots(json.dumps(raw), tier="long")
        self.assertTrue(ss.is_default("mood"))


class TestFillLoop(unittest.TestCase):
    def test_first_round_success(self):
        client = FakeClient([_payload_json()])
        result = fill_slots("屋顶花园画画的白发女孩", client, tier="long")
        self.assertEqual(result.rounds, 1)
        self.assertEqual(result.slots["setting"], "quiet rooftop garden")
        # system prompt + user intent, nothing else
        self.assertEqual(len(client.calls[0]), 2)
        self.assertIn("槽位", client.calls[0][0]["content"])

    def test_parse_error_triggers_repair_round(self):
        client = FakeClient([
            "这不是JSON",
            _payload_json(),
        ])
        result = fill_slots("需求", client, tier="long")
        self.assertEqual(result.rounds, 2)
        self.assertTrue(any("无法解析" in m["content"] for m in client.calls[1]))

    def test_consistency_error_triggers_repair_round(self):
        bad = rooftop_payload()
        bad["pose_family"] = "stand"   # stand + seated eye level = E_CAM_POSE
        client = FakeClient([
            json.dumps(bad, ensure_ascii=False),
            _payload_json(),
        ])
        result = fill_slots("需求", client, tier="long")
        self.assertEqual(result.rounds, 2)
        self.assertTrue(any("自洽问题" in m["content"] for m in client.calls[1]))

    def test_exhausted_rounds_return_last_state(self):
        bad = rooftop_payload()
        bad["pose_family"] = "stand"
        client = FakeClient([json.dumps(bad), json.dumps(bad), json.dumps(bad)])
        result = fill_slots("需求", client, tier="long", max_rounds=2)
        self.assertEqual(result.rounds, 3)
        self.assertTrue(result.feedback)


class TestSystemPrompt(unittest.TestCase):
    def test_tier_visibility_respected(self):
        long_sys = system_prompt("long")
        short_sys = system_prompt("short")
        self.assertIn("handheld", long_sys)
        self.assertNotIn("handheld ", short_sys)  # not among short-tier slots
        self.assertIn("pose_family", short_sys)

    def test_rules_present(self):
        sys_text = system_prompt("medium")
        for needle in ("src", "user", "inferred", "default", "JSON"):
            self.assertIn(needle, sys_text)


class TestResolveConfig(unittest.TestCase):
    def setUp(self):
        for var in ("OLIVIA_BASE_URL", "OLIVIA_API_KEY", "OLIVIA_MODEL"):
            os.environ.pop(var, None)

    def test_none_when_nothing_set(self):
        self.assertIsNone(resolve_config(config_path=Path("nonexistent.json")))

    def test_env_beats_file(self):
        os.environ["OLIVIA_BASE_URL"] = "http://127.0.0.1:8081"
        os.environ["OLIVIA_MODEL"] = "qwen3-4b"
        cfg = resolve_config(config_path=Path("nonexistent.json"))
        self.assertEqual(cfg.base_url, "http://127.0.0.1:8081/v1")
        self.assertEqual(cfg.model, "qwen3-4b")

    def test_explicit_args_beat_env(self):
        os.environ["OLIVIA_MODEL"] = "env-model"
        cfg = resolve_config(base_url="http://x:1/v1/", model="cli-model",
                             config_path=Path("nonexistent.json"))
        self.assertEqual(cfg.model, "cli-model")
        self.assertTrue(cfg.base_url.endswith("/v1"))

    def test_incomplete_config_is_none(self):
        cfg = resolve_config(base_url="http://x:1", model="",
                             config_path=Path("nonexistent.json"))
        self.assertIsNone(cfg)


class TestPipeline(unittest.TestCase):
    def test_full_pipeline(self):
        client = FakeClient([_payload_json()])
        result = run_intent("屋顶花园画画的白发女孩", tier="long",
                            ecosystem="illustrious", client=client)
        self.assertIn("long", result.prompts)
        self.assertIn("short", result.prompts)
        self.assertIn("medium", result.prompts)
        self.assertIn("multiple people", result.prompts["long"]["negative"])
        self.assertFalse([i for i in result.issues if i.severity == "error"])

    def test_cfg1_eco_patch_in_prompts(self):
        client = FakeClient([_payload_json()])
        result = run_intent("需求", tier="long", ecosystem="krea_turbo", client=client)
        block = result.prompts["long"]
        self.assertEqual(block["negative"], "")
        self.assertIn("a single subject only", block["positive_patch"])
        self.assertTrue(any("正向约束补丁" in n for n in result.notes))

    def test_empty_intent_rejected(self):
        client = FakeClient([])
        with self.assertRaises(ValueError):
            run_intent("  ", client=client)

    def test_no_client_and_no_config_raises_with_guidance(self):
        with self.assertRaises(RuntimeError) as ctx:
            run_intent("需求", client=None, config=None)
        self.assertIn("OLIVIA_BASE_URL", str(ctx.exception))


class TestChatClientWire(unittest.TestCase):
    """The real client against a local stub HTTP server."""

    def setUp(self):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                Handler.last_body = body  # record BEFORE responding (race-free)
                resp = json.dumps({
                    "choices": [{"message": {"content": _payload_json()}}],
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        Handler.last_body = None
        self.Handler = Handler

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_chat_roundtrip(self):
        cfg = ChatConfig(base_url=f"http://127.0.0.1:{self.port}/v1", model="stub")
        client = ChatClient(cfg)
        text = client.chat([{"role": "user", "content": "hi"}])
        self.assertIn("rooftop", text)
        # wire shape is OpenAI chat-completions
        self.assertEqual(self.Handler.last_body["model"], "stub")
        self.assertEqual(self.Handler.last_body["messages"][0]["role"], "user")

    def test_http_error_raises_chat_error(self):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class FailHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"boom")

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), FailHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            cfg = ChatConfig(base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
                             model="stub")
            with self.assertRaises(ChatError):
                ChatClient(cfg, retries=0).chat([{"role": "user", "content": "x"}])
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
