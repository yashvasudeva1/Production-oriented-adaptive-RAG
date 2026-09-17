from __future__ import annotations

from typing import List, Sequence

from ..ingestion.models import DocumentBlock
from .base import BaseChunker
from .models import Chunk


class ParentChildChunker(BaseChunker):
    """
    Generates hierarchical parent and child chunks.
    - Parents: broad context units (~1000-1500 tokens / sections)
    - Children: focused retrieval units (~200-300 tokens) pointing to parent_id.
    """

    def chunk(self, blocks: Sequence[DocumentBlock], document_id: str) -> List[Chunk]:
        all_chunks: List[Chunk] = []

        parent_chars = int(self.config.parent_chunk_size * self.config.chars_per_token)
        child_chars = int(self.config.child_chunk_size * self.config.chars_per_token)
        child_overlap = int(self.config.child_overlap * self.config.chars_per_token)

        for block_idx, block in enumerate(blocks):
            text = block.text.strip()
            if not text:
                continue

            # Split block text into parent segments
            parent_texts = self._split_into_parents(text, parent_chars)

            for p_idx, p_text in enumerate(parent_texts):
                parent_chunk_id = self.make_chunk_id(
                    document_id, f"p_b{block_idx}_{p_idx}"
                )
                parent_chunk = Chunk(
                    chunk_id=parent_chunk_id,
                    document_id=document_id,
                    parent_id=None,
                    text=p_text,
                    chunk_type="parent",
                    page=block.page,
                    section=block.section,
                    slide=block.slide,
                    sheet=block.sheet,
                    source_locator=block.source_locator or f"parent:{p_idx}",
                    token_count=self.estimate_tokens(p_text),
                    metadata={
                        **block.metadata,
                        "is_parent": True,
                    },
                )
                all_chunks.append(parent_chunk)

                # Now create child chunks inside this parent
                child_texts = self._split_into_children(p_text, child_chars, child_overlap)
                char_cursor = 0

                for c_idx, c_text in enumerate(child_texts):
                    start_char = p_text.find(c_text, char_cursor)
                    if start_char == -1:
                        start_char = char_cursor
                    end_char = start_char + len(c_text)
                    char_cursor = end_char

                    child_id = self.make_chunk_id(
                        document_id, f"c_{parent_chunk_id}_{c_idx}"
                    )
                    child_chunk = Chunk(
                        chunk_id=child_id,
                        document_id=document_id,
                        parent_id=parent_chunk_id,
                        text=c_text,
                        chunk_type="child",
                        page=block.page,
                        section=block.section,
                        slide=block.slide,
                        sheet=block.sheet,
                        source_locator=block.source_locator or f"child:{p_idx}:{c_idx}",
                        start_char=start_char,
                        end_char=end_char,
                        token_count=self.estimate_tokens(c_text),
                        metadata={
                            **block.metadata,
                            "parent_chunk_id": parent_chunk_id,
                            "is_child": True,
                        },
                    )
                    all_chunks.append(child_chunk)

        return all_chunks

    def _split_into_parents(self, text: str, max_chars: int) -> List[str]:
        if len(text) <= max_chars:
            return [text]
        paragraphs = text.split("\n\n")
        parents: List[str] = []
        current: List[str] = []
        current_len = 0

        for p in paragraphs:
            p_len = len(p)
            if current_len + p_len > max_chars and current:
                parents.append("\n\n".join(current))
                current = [p]
                current_len = p_len
            else:
                current.append(p)
                current_len += p_len

        if current:
            parents.append("\n\n".join(current))
        return parents

    def _split_into_children(self, text: str, child_chars: int, overlap: int) -> List[str]:
        if len(text) <= child_chars:
            return [text]

        children: List[str] = []
        start = 0
        while start < len(text):
            end = start + child_chars
            segment = text[start:end]
            if segment.strip():
                children.append(segment.strip())
            start = end - overlap
            if start >= len(text) - overlap:
                break
        return children
