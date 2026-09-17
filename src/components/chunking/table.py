from __future__ import annotations

import csv
import io
from typing import List, Sequence

from ..ingestion.models import DocumentBlock
from .base import BaseChunker
from .models import Chunk


class TableChunker(BaseChunker):
    """
    Table-aware chunker. Splits tabular data into groups of rows
    while ensuring table headers are repeated on every chunk so that
    retrieved rows are never severed from their schema headers.
    """

    def chunk(self, blocks: Sequence[DocumentBlock], document_id: str) -> List[Chunk]:
        chunks: List[Chunk] = []
        max_rows_per_chunk = 25  # rows per chunk

        for block_idx, block in enumerate(blocks):
            text = block.text.strip()
            if not text:
                continue

            lines = text.split("\n")
            if len(lines) <= 2:
                # Small table or 1 row
                chunk_id = self.make_chunk_id(document_id, f"tbl_b{block_idx}_0")
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        text=text,
                        chunk_type="table",
                        page=block.page,
                        section=block.section,
                        sheet=block.sheet,
                        source_locator=block.source_locator or f"table:sheet:{block.sheet or 'default'}",
                        token_count=self.estimate_tokens(text),
                        metadata={
                            **block.metadata,
                            "sheet_name": block.sheet,
                            "is_table": True,
                        },
                    )
                )
                continue

            header_line = lines[0]
            # If markdown table, separator line might be lines[1]
            has_md_separator = len(lines) > 1 and all(c in "-|: " for c in lines[1])
            data_start = 2 if has_md_separator else 1
            data_lines = lines[data_start:]

            for row_start in range(0, len(data_lines), max_rows_per_chunk):
                batch_rows = data_lines[row_start : row_start + max_rows_per_chunk]
                if has_md_separator:
                    chunk_text = "\n".join([header_line, lines[1]] + batch_rows)
                else:
                    chunk_text = "\n".join([header_line] + batch_rows)

                chunk_idx = row_start // max_rows_per_chunk
                chunk_id = self.make_chunk_id(
                    document_id, f"tbl_b{block_idx}_c{chunk_idx}"
                )

                row_from = row_start + 1
                row_to = min(row_start + max_rows_per_chunk, len(data_lines))
                locator = f"{block.source_locator or f'sheet:{block.sheet}'}:rows_{row_from}-{row_to}"

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        text=chunk_text,
                        chunk_type="table",
                        page=block.page,
                        section=block.section,
                        sheet=block.sheet,
                        source_locator=locator,
                        token_count=self.estimate_tokens(chunk_text),
                        metadata={
                            **block.metadata,
                            "sheet_name": block.sheet,
                            "is_table": True,
                            "rows_range": f"{row_from}-{row_to}",
                        },
                    )
                )

        return chunks
