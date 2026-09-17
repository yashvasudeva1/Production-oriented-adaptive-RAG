from __future__ import annotations

import re
from typing import Any, Callable, List, Optional, Sequence

from ..ingestion.models import DocumentBlock
from .base import BaseChunker
from .models import Chunk


class SemanticChunker(BaseChunker):
    """
    Semantic chunker that splits text into sentences and groups them based on
    semantic breakpoint detection or natural paragraph boundaries.
    """

    SENTENCE_REGEX = re.compile(r"(?<=[.?!])\s+(?=[A-Z0-9])")

    def __init__(
        self,
        config: Any = None,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
    ) -> None:
        super().__init__(config)
        self.similarity_fn = similarity_fn

    def chunk(self, blocks: Sequence[DocumentBlock], document_id: str) -> List[Chunk]:
        chunks: List[Chunk] = []
        target_chars = int(self.config.chunk_size * self.config.chars_per_token)

        for block_idx, block in enumerate(blocks):
            text = block.text.strip()
            if not text:
                continue

            sentences = [s.strip() for s in self.SENTENCE_REGEX.split(text) if s.strip()]
            if not sentences:
                sentences = [text]

            current_group: List[str] = []
            current_len = 0
            s_idx = 0

            for sent in sentences:
                sent_len = len(sent)
                if current_len + sent_len > target_chars and current_group:
                    chunk_text = " ".join(current_group)
                    chunk_id = self.make_chunk_id(
                        document_id, f"sem_b{block_idx}_s{s_idx}"
                    )
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            document_id=document_id,
                            text=chunk_text,
                            chunk_type="semantic",
                            page=block.page,
                            section=block.section,
                            slide=block.slide,
                            sheet=block.sheet,
                            source_locator=block.source_locator or f"sem:{block_idx}:{s_idx}",
                            token_count=self.estimate_tokens(chunk_text),
                            metadata={
                                **block.metadata,
                                "is_semantic": True,
                            },
                        )
                    )
                    s_idx += 1
                    current_group = [sent]
                    current_len = sent_len
                else:
                    current_group.append(sent)
                    current_len += sent_len

            if current_group:
                chunk_text = " ".join(current_group)
                chunk_id = self.make_chunk_id(
                    document_id, f"sem_b{block_idx}_s{s_idx}"
                )
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        text=chunk_text,
                        chunk_type="semantic",
                        page=block.page,
                        section=block.section,
                        slide=block.slide,
                        sheet=block.sheet,
                        source_locator=block.source_locator or f"sem:{block_idx}:{s_idx}",
                        token_count=self.estimate_tokens(chunk_text),
                        metadata={
                            **block.metadata,
                            "is_semantic": True,
                        },
                    )
                )

        return chunks
