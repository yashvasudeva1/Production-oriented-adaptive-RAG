from __future__ import annotations

import re
from typing import List, Sequence

from ..ingestion.models import DocumentBlock
from .base import BaseChunker
from .models import Chunk


class CodeChunker(BaseChunker):
    """
    Code-aware chunker that identifies functions, classes, and top-level
    blocks, preserving scope headers and indentation context.
    """

    # Matches class and def in Python, function/export in JS/TS, etc.
    DEF_REGEX = re.compile(
        r"^(?:async\s+)?(?:def|class|function|export\s+(?:default\s+)?(?:class|function))\s+([a-zA-Z_0-9]+)",
        re.MULTILINE,
    )

    def chunk(self, blocks: Sequence[DocumentBlock], document_id: str) -> List[Chunk]:
        chunks: List[Chunk] = []
        target_chars = int(self.config.chunk_size * self.config.chars_per_token)

        for block_idx, block in enumerate(blocks):
            code_text = block.text
            if not code_text.strip():
                continue

            lines = code_text.split("\n")
            symbols: List[tuple[str, int, int]] = []  # (symbol_name, start_line, end_line)

            for i, line in enumerate(lines):
                match = self.DEF_REGEX.match(line)
                if match:
                    symbols.append((match.group(0).strip(), i, len(lines)))

            # If no definitions or single short definition, emit as single chunk
            if not symbols or (len(symbols) == 1 and len(code_text) <= target_chars):
                chunk_id = self.make_chunk_id(document_id, f"code_b{block_idx}_0")
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        text=code_text.strip(),
                        chunk_type="code",
                        page=block.page,
                        section=block.section,
                        source_locator=block.source_locator or f"code:block:{block_idx}",
                        token_count=self.estimate_tokens(code_text),
                        metadata={
                            **block.metadata,
                            "is_code": True,
                        },
                    )
                )
                continue

            # Chunk by identified functions/classes
            for sym_idx in range(len(symbols)):
                name, start_l, _ = symbols[sym_idx]
                end_l = symbols[sym_idx + 1][1] if sym_idx + 1 < len(symbols) else len(lines)

                snippet = "\n".join(lines[start_l:end_l]).strip()
                if not snippet:
                    continue

                chunk_id = self.make_chunk_id(
                    document_id, f"code_b{block_idx}_{sym_idx}"
                )
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        text=snippet,
                        chunk_type="code",
                        page=block.page,
                        section=name,
                        source_locator=f"{block.source_locator or f'code:{block_idx}'}:lines_{start_l+1}-{end_l}",
                        token_count=self.estimate_tokens(snippet),
                        metadata={
                            **block.metadata,
                            "symbol": name,
                            "is_code": True,
                            "line_start": start_l + 1,
                            "line_end": end_l,
                        },
                    )
                )

        return chunks
