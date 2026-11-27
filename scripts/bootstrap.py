"""One-shot bootstrap for the double-click experience (M5).

start.bat -> this script.  Four stages, each idempotent:

1. package     — pip install -e . when olivia-prompt is missing
2. weights     — read models/models.json, download the chosen GGUF (resumable
                 via huggingface_hub), verify sha256
3. llama.cpp   — download the prebuilt Windows CUDA llama-server zip once
4. serve       — llama-server on 127.0.0.1:8081, write ~/.olivia-prompt/
                 config.json, then `olivia-prompt web --open`

Everything lands under OLIVIA_MODELS_DIR (default ./models) so the heavy files
stay out of the repo.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_JSON = ROOT / "models" / "models.json"
LLAMA_RELEASE = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{asset}"
LLAMA_TAG = "b12606"
LLAMA_ASSET = "llama-b12606-bin-win-cuda-12.4-x64.zip"
CUDART_ASSET = "cudart-llama-bin-win-cuda-12.4-x64.zip"


def log(msg: str) -> None:
    print(f"[start] {msg}", flush=True)


def ensure_package() -> None:
    try:
        import olivia_prompt  # noqa: F401
        return
    except ImportError:
        pass
    log("安装 olivia-prompt（pip install -e .）...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(ROOT)])


def load_manifest() -> dict:
    return json.loads(MODELS_JSON.read_text(encoding="utf-8"))


def download_weights(key: str, models_dir: Path) -> Path:
    manifest = load_manifest()
    entry = manifest["models"][key]
    local = models_dir / entry["file"]
    if local.exists() and _sha256_ok(local, entry.get("sha256")):
        log(f"权重已就绪: {local.name}")
        return local

    models_dir.mkdir(parents=True, exist_ok=True)
    url = entry["url"]
    log(f"下载权重 {key}: {url}")
    if url.startswith("hf://"):
        repo_path = url[len("hf://"):]
        repo, _, filename = repo_path.partition("/")
        from huggingface_hub import hf_hub_download
        got = hf_hub_download(repo_id=repo, filename=filename, local_dir=models_dir)
        if Path(got).resolve() != local.resolve():
            shutil.copy2(got, local)
    else:
        _curl(url, local)
    if entry.get("sha256"):
        _require_sha256(local, entry["sha256"])
    return local


def _curl(url: str, dest: Path) -> None:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "OliviaPrompt/0.2"})
    with urllib.request.urlopen(req) as resp, dest.open("wb") as fh:
        shutil.copyfileobj(resp, fh, 1 << 20)


def _sha256_ok(path: Path, expected: str | None) -> bool:
    if not expected:
        return True
    return _sha256(path) == expected


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_sha256(path: Path, expected: str) -> None:
    got = _sha256(path)
    if got != expected:
        path.unlink(missing_ok=True)
        raise SystemExit(f"sha256 校验失败: {path.name}\n  got      {got}\n  expected {expected}")


def ensure_llama_server(tools_dir: Path) -> Path:
    exe = tools_dir / "llama-server.exe"
    if exe.exists():
        return exe
    tools_dir.mkdir(parents=True, exist_ok=True)
    log("下载 llama.cpp 预编译 llama-server（首次约 300MB）...")
    for asset in (LLAMA_ASSET, CUDART_ASSET):
        dest = tools_dir / asset
        if not dest.exists():
            _curl(LLAMA_RELEASE.format(tag=LLAMA_TAG, asset=asset), dest)
        with zipfile.ZipFile(dest) as zf:
            zf.extractall(tools_dir)
    if not exe.exists():
        raise SystemExit(f"llama-server 解压后不存在: {exe}")
    log(f"llama-server 就绪: {exe}")
    return exe


def write_client_config(models_dir: Path, model_file: str) -> None:
    cfg_dir = Path.home() / ".olivia-prompt"
    cfg_dir.mkdir(exist_ok=True)
    cfg = {"base_url": "http://127.0.0.1:8081/v1", "model": "olivia"}
    (cfg_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    log(f"客户端配置 -> {cfg_dir / 'config.json'}")


def main(argv: list[str]) -> int:
    key = argv[0] if argv else "lite"
    models_dir = Path(os.environ.get("OLIVIA_MODELS_DIR", ROOT / "models"))
    tools_dir = models_dir.parent / "tools" / "llama.cpp"

    ensure_package()
    gguf = download_weights(key, models_dir)
    server = ensure_llama_server(tools_dir)
    write_client_config(models_dir, gguf.name)

    import webbrowser
    log(f"启动 llama-server: {gguf.name}")
    llama_proc = subprocess.Popen(
        [str(server), "-m", str(gguf), "--port", "8081", "--host", "127.0.0.1",
         "-ngl", "999", "--parallel", "2"],
        cwd=str(tools_dir),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    log("启动 Web UI (http://127.0.0.1:7860)")
    web_proc = subprocess.Popen(
        [sys.executable, "-m", "olivia_prompt.cli", "web", "--bind", "127.0.0.1:7860", "--open"],
        cwd=str(ROOT),
    )
    try:
        webbrowser.open("http://127.0.0.1:7860")
    except OSError:
        pass
    log("就绪。Ctrl+C 退出。")
    try:
        web_proc.wait()
    finally:
        llama_proc.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
