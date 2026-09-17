from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..base import BaseLoader
from ..models import DocumentBlock


TEXT_EXTENSIONS = (
    ".txt", ".text", ".log", ".rtf",
)

MARKDOWN_EXTENSIONS = (
    ".md", ".markdown", ".mdown", ".mkdn",
)


class PlainTextLoader(BaseLoader):
    name = "plain-text"
    extensions = TEXT_EXTENSIONS

    def load_blocks(self, path: Path) -> Iterable[DocumentBlock]:
        text = _read_text(path)
        yield DocumentBlock(
            block_id="",
            text=text,
            block_type="text",
            source_locator=str(path.name),
        )


class MarkdownLoader(BaseLoader):
    name = "markdown"
    extensions = MARKDOWN_EXTENSIONS

    def load_blocks(self, path: Path) -> Iterable[DocumentBlock]:
        text = _read_text(path)
        current_section = None
        buffer: list[str] = []
        in_code = False

        def flush():
            nonlocal buffer
            if buffer:
                payload = "\n".join(buffer).strip()
                if payload:
                    yield DocumentBlock(
                        block_id="",
                        text=payload,
                        block_type="code" if in_code else "markdown",
                        section=current_section,
                        source_locator=str(path.name),
                    )
                buffer = []

        for line in text.splitlines():
            if line.strip().startswith("```"):
                yield from flush()
                in_code = not in_code
                continue

            if not in_code and line.startswith("#"):
                yield from flush()
                current_section = line.lstrip("#").strip()
                yield DocumentBlock(
                    block_id="",
                    text=current_section,
                    block_type="heading",
                    section=current_section,
                    source_locator=str(path.name),
                )
                continue

            buffer.append(line)

        yield from flush()


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")
