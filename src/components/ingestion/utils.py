from __future__ import annotations

import hashlib
import mimetypes
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Optional

from .models import DocumentBlock


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def detect_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    return mime or "application/octet-stream"


def make_block_id(document_hash: str, index: int) -> str:
    return f"{document_hash[:16]}-b{index:05d}"


def normalize_text(text: str, preserve_whitespace: bool = False) -> str:
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)

    if preserve_whitespace:
        # Only remove pathological blank-line runs.
        return re.sub(r"\n{4,}", "\n\n\n", text).strip()

    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def nonempty_blocks(blocks: Iterable[DocumentBlock]) -> list[DocumentBlock]:
    return [b for b in blocks if b.text and b.text.strip()]


def safe_member_name(name: str) -> Optional[Path]:
    # Prevent path traversal when dealing with archives.
    candidate = Path(name)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    return candidate
