"""OpenAI-compatible chat client, stdlib only.

Zero dependencies is a product feature here: the whole point of OliviaPrompt
is "download, double-click, use".  ``urllib`` is enough for /chat/completions,
and retrying a flaky localhost connection needs nothing fancier.

Config resolution order (first hit wins):

1. explicit arguments (CLI flags, web UI request body)
2. environment variables ``OLIVIA_BASE_URL`` / ``OLIVIA_API_KEY`` / ``OLIVIA_MODEL``
3. ``~/.olivia-prompt/config.json`` with keys ``base_url`` / ``api_key`` / ``model``

A config is only "ready" when base_url *and* model are both present; api_key is
optional (llama-server doesn't want one).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

ENV_BASE_URL = "OLIVIA_BASE_URL"
ENV_API_KEY = "OLIVIA_API_KEY"
ENV_MODEL = "OLIVIA_MODEL"

CONFIG_FILE = Path.home() / ".olivia-prompt" / "config.json"

DEFAULT_TIMEOUT = 180.0
DEFAULT_MAX_TOKENS = 2048


class ChatError(RuntimeError):
    """Raised when the endpoint cannot be reached or answers with an error."""


@dataclass(frozen=True)
class ChatConfig:
    base_url: str
    model: str
    api_key: str = ""
    temperature: float = 0.6
    timeout: float = DEFAULT_TIMEOUT
    max_tokens: int = DEFAULT_MAX_TOKENS

    def ready(self) -> bool:
        return bool(self.base_url and self.model)


def _normalize_base_url(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    if url and not url.endswith("/v1") and "/v" not in url.rsplit("/", 1)[-1]:
        # accept both http://host:port and http://host:port/v1
        url = url + "/v1"
    return url


def resolve_config(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    config_path: Optional[Path] = None,
) -> Optional[ChatConfig]:
    """Resolve connection settings; None when nothing is configured."""
    base_url = base_url or os.environ.get(ENV_BASE_URL, "")
    api_key = api_key or os.environ.get(ENV_API_KEY, "")
    model = model or os.environ.get(ENV_MODEL, "")

    path = Path(config_path) if config_path else CONFIG_FILE
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
        base_url = base_url or str(data.get("base_url") or "")
        api_key = api_key or str(data.get("api_key") or "")
        model = model or str(data.get("model") or "")

    base_url = _normalize_base_url(base_url)
    if not base_url or not model:
        return None
    return ChatConfig(base_url=base_url, model=model, api_key=api_key)


class ChatClient:
    """Minimal OpenAI-compatible chat client with bounded retries."""

    def __init__(self, config: ChatConfig, retries: int = 2):
        self.config = config
        self.retries = retries

    # -- public -------------------------------------------------------------
    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        cfg = self.config
        payload: Dict[str, object] = {
            "model": cfg.model,
            "messages": messages,
            "temperature": cfg.temperature if temperature is None else temperature,
            "max_tokens": cfg.max_tokens if max_tokens is None else max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(min(2.0 * attempt, 4.0))
            try:
                return self._post(payload)
            except ChatError as exc:
                last_error = exc
        raise last_error  # type: ignore[misc]

    def describe(self) -> str:
        return f"{self.config.base_url}  model={self.config.model}"

    # -- internals ----------------------------------------------------------
    def _post(self, payload: Dict[str, object]) -> str:
        cfg = self.config
        url = f"{cfg.base_url}/chat/completions"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:400]
            except OSError:
                pass
            raise ChatError(f"HTTP {exc.code} from {url}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ChatError(f"cannot reach {url}: {exc}") from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ChatError(f"unexpected response shape from {url}: {data}") from exc
        if content is None:
            raise ChatError(f"empty completion from {url}")
        return str(content)


__all__ = [
    "ChatClient",
    "ChatConfig",
    "ChatError",
    "resolve_config",
    "CONFIG_FILE",
    "ENV_BASE_URL",
    "ENV_API_KEY",
    "ENV_MODEL",
]
