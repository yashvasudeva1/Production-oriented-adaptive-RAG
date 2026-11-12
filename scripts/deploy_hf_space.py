"""Deploy Nexus to a free Hugging Face Gradio Space following best practices.

Best practices enforced:
  - Never uploads .env, .git, .venv, __pycache__, or frontend/node_modules.
  - Injects GROQ_API_KEY and QDRANT_API_KEY via HF Space Secrets API (never in repo files).
  - Generates a Space-specific README.md (sdk: gradio, app_file: app.py) and
    trimmed requirements.txt (avoids heavy eval/CUDA/audio deps conflicting with Gradio).

Usage:
    # 1. Authenticate once with a HF write token:
    #    .venv/Scripts/python.exe -m huggingface_hub.commands.huggingface_cli login
    # 2. Dry-run to inspect staged files:
    #    python scripts/deploy_hf_space.py --repo-id <hf-user>/nexus-adaptive-rag --dry-run
    # 3. Deploy + sync secrets/variables from .env:
    #    python scripts/deploy_hf_space.py --repo-id <hf-user>/nexus-adaptive-rag
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict

from dotenv import dotenv_values

ROOT_DIR = Path(__file__).resolve().parents[1]

SPACE_README = """---
title: Adaptive Rag
emoji: 🔥
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
python_version: '3.12'
app_file: app.py
pinned: false
---

# Nexus — Production-Oriented Adaptive RAG

- **Gradio UI:** `/`
- **HTML SSE Demo:** `/demo`
- **Swagger API Docs:** `/docs`
- **Cron Health Check:** `/health` (`GET` & `HEAD`)
- **Prometheus Metrics:** `/metrics`
"""

SPACE_REQUIREMENTS = """spaces
pydantic-settings
python-dotenv
python-multipart
requests
qdrant-client
rank-bm25
sentence-transformers
transformers
huggingface-hub
numpy
groq
langchain-core
langchain-groq
langsmith
pymupdf
pypdf
python-docx
python-pptx
openpyxl
beautifulsoup4
pyyaml
pillow
"""

INCLUDE_FILES = ("app.py",)
INCLUDE_DIRS = ("src", "config", "prompts", "demo")
IGNORE_PATTERNS = shutil.ignore_patterns(
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".pytest_cache",
    ".git",
    ".venv",
    ".env",
)


def stage_space_bundle(dest: Path) -> None:
    (dest / "README.md").write_text(SPACE_README, encoding="utf-8")
    (dest / "requirements.txt").write_text(SPACE_REQUIREMENTS, encoding="utf-8")

    for rel in INCLUDE_FILES:
        src = ROOT_DIR / rel
        if src.exists():
            shutil.copy2(src, dest / rel)

    for d in INCLUDE_DIRS:
        src_dir = ROOT_DIR / d
        if src_dir.exists():
            shutil.copytree(src_dir, dest / d, ignore=IGNORE_PATTERNS)

    # Include metadata folder without overwriting runtime secrets; ensure chunks.json exists
    meta_dest = dest / "metadata"
    meta_dest.mkdir(parents=True, exist_ok=True)
    for item in ("chunks.json", "documents.json"):
        src_item = ROOT_DIR / "metadata" / item
        if src_item.exists():
            shutil.copy2(src_item, meta_dest / item)
    if not (meta_dest / "chunks.json").exists():
        (meta_dest / "chunks.json").write_text("[]", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage and push Nexus to HF Gradio Space.")
    parser.add_argument("--repo-id", required=True, help="Target HF Space repo, e.g. username/nexus-adaptive-rag")
    parser.add_argument("--private", action="store_true", help="Create Space as private (default: public).")
    parser.add_argument("--dry-run", action="store_true", help="Stage files locally without uploading.")
    args = parser.parse_args()

    env_vals: Dict[str, str | None] = dotenv_values(ROOT_DIR / ".env") if (ROOT_DIR / ".env").exists() else {}

    with tempfile.TemporaryDirectory(prefix="hf_space_") as tmp:
        stage_dir = Path(tmp)
        stage_space_bundle(stage_dir)

        staged_files = sorted(str(p.relative_to(stage_dir)) for p in stage_dir.rglob("*") if p.is_file())
        assert ".env" not in staged_files, "Refusing to stage .env file!"

        if args.dry_run:
            print(f"Dry-run staged {len(staged_files)} files for '{args.repo_id}' (no secrets included):")
            for f in staged_files[:25]:
                print(" ", f)
            if len(staged_files) > 25:
                print(f"  ... and {len(staged_files) - 25} more files")
            return

        from huggingface_hub import HfApi

        token = os.getenv("HF_TOKEN") or env_vals.get("HF_TOKEN") or None
        api = HfApi(token=token)
        who = api.whoami()
        print(f"Authenticated to Hugging Face as: {who.get('name')}")

        api.create_repo(
            repo_id=args.repo_id,
            repo_type="space",
            space_sdk="gradio",
            private=args.private,
            exist_ok=True,
        )

        # Configure secrets & non-sensitive runtime variables
        for secret_key in ("GROQ_API_KEY", "QDRANT_API_KEY", "OPENROUTER_API_KEY"):
            val = os.getenv(secret_key) or env_vals.get(secret_key)
            if val:
                api.add_space_secret(repo_id=args.repo_id, key=secret_key, value=val)

        space_vars = {
            "QDRANT_URL": os.getenv("QDRANT_URL") or env_vals.get("QDRANT_URL") or "",
            "QDRANT_COLLECTION": env_vals.get("QDRANT_COLLECTION") or "nexus_chunks",
            "LLM_PROVIDER": env_vals.get("LLM_PROVIDER") or "groq",
            "LLM_MODEL": env_vals.get("LLM_MODEL") or "qwen/qwen3.8-27b",
            "EMBEDDING_MODEL": env_vals.get("EMBEDDING_MODEL") or "thenlper/gte-small",
            "RERANKER_MODEL": env_vals.get("RERANKER_MODEL") or "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "TRANSCRIBE_AUDIO_VIDEO": "false",
            "OFFLINE_EVAL": "false",
        }
        for k, v in space_vars.items():
            if v:
                api.add_space_variable(repo_id=args.repo_id, key=k, value=v)

        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="space",
            folder_path=str(stage_dir),
            commit_message="Deploy Nexus Adaptive RAG (Gradio + FastAPI)",
        )
        print(f"Deployed successfully: https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
