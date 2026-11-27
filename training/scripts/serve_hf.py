"""Tiny OpenAI-compatible server over transformers+peft — the eval harness's
GPU serving path (llama.cpp binaries are only needed for the final GGUF
smoke test, and the release CDN is slow on some networks).

    python training/scripts/serve_hf.py \
        --model E:/OliviaPrompt/models/Qwen3-4B \
        [--adapter training/out/lite-4b] \
        [--load-in-4bit] [--port 8082]

Endpoints: POST /v1/chat/completions (non-streaming).  Qwen3 thinking is
disabled to match the SFT contract (no <think> blocks in training targets).
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch


class Engine:
    def __init__(self, model: str, adapter: str | None, load_in_4bit: bool):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model)
        kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            kwargs["torch_dtype"] = None
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
            print(f"[serve_hf] adapter loaded: {adapter}", flush=True)
        self.model.eval()
        self.lock = threading.Lock()

    def chat(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        do_sample = temperature and temperature > 0.05
        with self.lock:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    temperature=max(temperature, 1e-3) if do_sample else None,
                    top_p=0.9 if do_sample else None,
                    pad_token_id=self.tok.eos_token_id,
                )
        text = self.tok.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        # qwen3 may still open a think block; the harness strips it too, but
        # keep the wire clean here
        if "</think>" in text:
            text = text.split("</think>", 1)[-1]
        return text.strip()


class Handler(BaseHTTPRequestHandler):
    engine: Engine = None  # type: ignore[assignment]

    def log_message(self, *args):
        pass

    def do_POST(self):
        if not self.path.endswith("/chat/completions"):
            self._json(404, {"error": f"no route {self.path}"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        t0 = time.time()
        try:
            text = self.engine.chat(
                body["messages"],
                temperature=float(body.get("temperature") or 0.2),
                max_tokens=int(body.get("max_tokens") or 1024),
            )
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"error": str(exc)[:300]})
            return
        self._json(200, {
            "id": "serve-hf",
            "object": "chat.completion",
            "model": body.get("model", "local"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"latency_s": round(time.time() - t0, 2)},
        })

    def _json(self, status: int, data: dict) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    print(f"[serve_hf] loading {args.model}" +
          (f" + adapter {args.adapter}" if args.adapter else "") +
          (" (4bit)" if args.load_in_4bit else ""), flush=True)
    Handler.engine = Engine(args.model, args.adapter, args.load_in_4bit)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[serve_hf] ready on http://127.0.0.1:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
