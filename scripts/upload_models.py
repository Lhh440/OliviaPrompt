"""Upload the produced GGUFs to Hugging Face (M5 分发).

Weight files never live in the repo.  models/models.json points at a HF repo
with resumable downloads + sha256 verification; this helper pushes them there.

    set HF_WRITE_TOKEN=hf_xxx
    set HF_REPO=Lhh440/OliviaPrompt-GGUF
    python scripts/upload_models.py --dir E:/OliviaPrompt/models

Uploads every *.gguf found in --dir, plus sha256 manifest regeneration:
    python scripts/upload_models.py --manifest-only --dir E:/OliviaPrompt/models
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "models"))
    ap.add_argument("--repo", default="Lhh440/OliviaPrompt-GGUF")
    ap.add_argument("--manifest-only", action="store_true",
                    help="只把本地 sha256 写回 models/models.json，不上传")
    args = ap.parse_args(argv)

    models_dir = Path(args.dir)
    ggufs = sorted(models_dir.glob("*.gguf"))
    if not ggufs:
        raise SystemExit(f"{models_dir} 下没有 *.gguf")

    manifest_path = ROOT / "models" / "models.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # refresh sha256 for known entries
    for key, entry in manifest["models"].items():
        local = models_dir / entry["file"]
        if local.exists():
            entry["sha256"] = sha256(local)
            entry.setdefault("url", f"hf://{args.repo}/{entry['file']}")
            print(f"[manifest] {key}: {entry['file']} sha256={entry['sha256'][:16]}…")

    manifest["meta"]["updated"] = manifest["meta"].get("updated", "")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(f"[manifest] 写回 {manifest_path}")

    if args.manifest_only:
        return 0

    token = __import__("os").environ.get("HF_WRITE_TOKEN")
    if not token:
        raise SystemExit(
            "缺少 HF_WRITE_TOKEN。到 https://huggingface.co/settings/tokens 取 write token 后:\n"
            "  set HF_WRITE_TOKEN=hf_xxx\n"
            "  python scripts/upload_models.py --dir <GGUF目录>\n"
            "（也可手动上传，文件名须与 models/models.json 的 file 字段一致）"
        )
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(args.repo, exist_ok=True, repo_type="model")
    for gg in ggufs:
        print(f"[upload] {gg.name} ...")
        api.upload_file(path_or_fileobj=str(gg), repo_id=args.repo,
                        path_in_repo=gg.name, repo_type="model")
    print("上传完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
