from __future__ import annotations

from typing import List, Sequence

from ..ingestion.models import DocumentBlock
from .base import BaseChunker
from .models import Chunk


class RecursiveChunker(BaseChunker):
    """
    Splits text recursively using a hierarchy of separators,
    preserving block metadata and provenance coordinates.
    """

    def chunk(self, blocks: Sequence[DocumentBlock], document_id: str) -> List[Chunk]:
        chunks: List[Chunk] = []

        for block_idx, block in enumerate(blocks):
            text = block.text.strip()
            if not text:
                continue

            block_chunks = self._split_text(text)
            char_offset = 0

            for sub_idx, sub_text in enumerate(block_chunks):
                if not sub_text.strip():
                    continue

                token_count = self.estimate_tokens(sub_text)
                start_char = text.find(sub_text, char_offset)
                if start_char == -1:
                    start_char = char_offset
                end_char = start_char + len(sub_text)
                char_offset = end_char

                chunk_id = self.make_chunk_id(
                    document_id,
                    f"b{block_idx}_{block.block_id or 'blk'}_s{sub_idx}",
                )

                chunk = Chunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    text=sub_text.strip(),
                    chunk_type=block.block_type or "text",
                    page=block.page,
                    section=block.section,
                    slide=block.slide,
                    sheet=block.sheet,
                    source_locator=block.source_locator or f"block:{block_idx}",
                    start_char=start_char,
                    end_char=end_char,
                    token_count=token_count,
                    metadata=dict(block.metadata),
                )
                chunks.append(chunk)

        return chunks

    def _split_text(self, text: str) -> List[str]:
        target_chars = int(self.config.chunk_size * self.config.chars_per_token)
        overlap_chars = int(self.config.chunk_overlap * self.config.chars_per_token)

        if len(text) <= target_chars:
            return [text]

        return self._recursive_split(
            text, self.config.separators, target_chars, overlap_chars
        )

    def _recursive_split(
        self,
        text: str,
        separators: List[str],
        target_chars: int,
        overlap_chars: int,
    ) -> List[str]:
        if len(text) <= target_chars:
            return [text]

        if not separators:
            # Fallback to hard character slicing
            results = []
            start = 0
            while start < len(text):
                end = start + target_chars
                results.append(text[start:end])
                start = end - overlap_chars
                if start >= len(text) - overlap_chars:
                    break
            return results

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            return [text[i : i + target_chars] for i in range(0, len(text), target_chars - overlap_chars)]

        splits = text.split(sep) if sep in text else [text]
        if len(splits) == 1:
            return self._recursive_split(text, remaining_seps, target_chars, overlap_chars)

        docs: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for s in splits:
            item = s + sep
            item_len = len(item)

            if item_len > target_chars:
                if current_chunk:
                    docs.append("".join(current_chunk).strip())
                    current_chunk = []
                    current_len = 0
                sub_splits = self._recursive_split(s, remaining_seps, target_chars, overlap_chars)
                docs.extend(sub_splits)
                continue

            if current_len + item_len > target_chars:
                if current_chunk:
                    docs.append("".join(current_chunk).strip())
                current_chunk = [item]
                current_len = item_len
            else:
                current_chunk.append(item)
                current_len += item_len

        if current_chunk:
            remaining_text = "".join(current_chunk).strip()
            if remaining_text:
                docs.append(remaining_text)

        # Merge undersized chunks or add overlap
        return [d for d in docs if len(d.strip()) > 0]
