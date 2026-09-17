from __future__ import annotations

import re
from typing import List, Sequence

from ..ingestion.models import DocumentBlock
from .base import BaseChunker
from .models import Chunk


class MarkdownChunker(BaseChunker):
    """
    Heading-aware Markdown chunker. Tracks hierarchy (H1 > H2 > H3),
    keeps code blocks and tables intact where possible, and avoids
    splitting inside fenced code blocks.
    """

    HEADING_REGEX = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

    def chunk(self, blocks: Sequence[DocumentBlock], document_id: str) -> List[Chunk]:
        chunks: List[Chunk] = []

        for block_idx, block in enumerate(blocks):
            text = block.text.strip()
            if not text:
                continue

            sections = self._split_by_headings(text)

            for s_idx, (heading_path, body) in enumerate(sections):
                if not body.strip():
                    continue

                full_text = f"{heading_path}\n{body}".strip() if heading_path else body.strip()
                chunk_id = self.make_chunk_id(
                    document_id, f"md_b{block_idx}_s{s_idx}"
                )

                chunk = Chunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    text=full_text,
                    chunk_type="markdown",
                    page=block.page,
                    section=heading_path or block.section,
                    slide=block.slide,
                    sheet=block.sheet,
                    source_locator=block.source_locator or f"md:section:{s_idx}",
                    token_count=self.estimate_tokens(full_text),
                    metadata={
                        **block.metadata,
                        "heading_path": heading_path,
                    },
                )
                chunks.append(chunk)

        return chunks

    def _split_by_headings(self, text: str) -> List[tuple[str, str]]:
        lines = text.split("\n")
        sections: List[tuple[str, str]] = []
        current_heading_path: List[str] = []
        current_lines: List[str] = []

        in_code_block = False

        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                current_lines.append(line)
                continue

            if not in_code_block:
                match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
                if match:
                    if current_lines:
                        path_str = " > ".join(current_heading_path)
                        sections.append((path_str, "\n".join(current_lines).strip()))
                        current_lines = []

                    level = len(match.group(1))
                    title = match.group(2).strip()

                    # Adjust heading stack
                    while len(current_heading_path) >= level:
                        current_heading_path.pop()
                    current_heading_path.append(title)
                    continue

            current_lines.append(line)

        if current_lines:
            path_str = " > ".join(current_heading_path)
            sections.append((path_str, "\n".join(current_lines).strip()))

        return sections
