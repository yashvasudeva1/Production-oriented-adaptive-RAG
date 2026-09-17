from __future__ import annotations

import re
from typing import List, Sequence

from ..ingestion.models import DocumentBlock
from .base import BaseChunker
from .models import Chunk


class TranscriptChunker(BaseChunker):
    """
    Time-aware chunker for audio and video transcripts.
    Preserves start and end timestamps in source locators and metadata.
    """

    TIMESTAMP_REGEX = re.compile(r"\[?(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)\]?")

    def chunk(self, blocks: Sequence[DocumentBlock], document_id: str) -> List[Chunk]:
        chunks: List[Chunk] = []
        target_chars = int(self.config.chunk_size * self.config.chars_per_token)

        for block_idx, block in enumerate(blocks):
            text = block.text.strip()
            if not text:
                continue

            lines = [line.strip() for line in text.split("\n") if line.strip()]
            current_batch: List[str] = []
            current_len = 0
            start_ts = None
            end_ts = None
            c_idx = 0

            for line in lines:
                ts_match = self.TIMESTAMP_REGEX.search(line)
                if ts_match:
                    found_ts = ts_match.group(1)
                    if start_ts is None:
                        start_ts = found_ts
                    end_ts = found_ts

                if current_len + len(line) > target_chars and current_batch:
                    chunk_text = "\n".join(current_batch)
                    chunk_id = self.make_chunk_id(
                        document_id, f"transcript_b{block_idx}_c{c_idx}"
                    )
                    time_locator = (
                        f"time:{start_ts or '00:00'}-{end_ts or start_ts or '00:00'}"
                    )

                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            document_id=document_id,
                            text=chunk_text,
                            chunk_type="transcript",
                            source_locator=block.source_locator or time_locator,
                            token_count=self.estimate_tokens(chunk_text),
                            metadata={
                                **block.metadata,
                                "start_time": start_ts,
                                "end_time": end_ts,
                                "is_transcript": True,
                            },
                        )
                    )
                    c_idx += 1
                    current_batch = [line]
                    current_len = len(line)
                    start_ts = ts_match.group(1) if ts_match else None
                else:
                    current_batch.append(line)
                    current_len += len(line)

            if current_batch:
                chunk_text = "\n".join(current_batch)
                chunk_id = self.make_chunk_id(
                    document_id, f"transcript_b{block_idx}_c{c_idx}"
                )
                time_locator = (
                    f"time:{start_ts or '00:00'}-{end_ts or start_ts or '00:00'}"
                )
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        text=chunk_text,
                        chunk_type="transcript",
                        source_locator=block.source_locator or time_locator,
                        token_count=self.estimate_tokens(chunk_text),
                        metadata={
                            **block.metadata,
                            "start_time": start_ts,
                            "end_time": end_ts,
                            "is_transcript": True,
                        },
                    )
                )

        return chunks
