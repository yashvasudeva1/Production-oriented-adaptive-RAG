from __future__ import annotations

import re
from typing import List, Sequence

from ..ingestion.models import DocumentBlock
from .base import BaseChunker
from .models import Chunk


class StructuralChunker(BaseChunker):
    """
    Structure-aware chunker that understands document sections, headings,
    and hierarchical structure. Never severs paragraphs abruptly and
    retains section title context on sub-chunks.
    """

    def chunk(self, blocks: Sequence[DocumentBlock], document_id: str) -> List[Chunk]:
        chunks: List[Chunk] = []
        target_chars = int(self.config.chunk_size * self.config.chars_per_token)

        for block_idx, block in enumerate(blocks):
            text = block.text.strip()
            if not text:
                continue

            section_title = block.section or block.metadata.get("section", "")
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

            if not paragraphs:
                paragraphs = [text]

            current_paras: List[str] = []
            current_len = 0
            sub_idx = 0

            for para in paragraphs:
                para_len = len(para)
                if current_len + para_len > target_chars and current_paras:
                    # Flush current group
                    combined_text = "\n\n".join(current_paras)
                    if section_title and not combined_text.startswith(section_title):
                        chunk_text = f"[{section_title}]\n{combined_text}"
                    else:
                        chunk_text = combined_text

                    chunk_id = self.make_chunk_id(
                        document_id, f"struct_b{block_idx}_s{sub_idx}"
                    )
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            document_id=document_id,
                            text=chunk_text,
                            chunk_type="section" if section_title else (block.block_type or "text"),
                            page=block.page,
                            section=section_title or None,
                            slide=block.slide,
                            sheet=block.sheet,
                            source_locator=block.source_locator or f"section:{section_title or block_idx}",
                            token_count=self.estimate_tokens(chunk_text),
                            metadata={
                                **block.metadata,
                                "section_title": section_title,
                            },
                        )
                    )
                    sub_idx += 1
                    current_paras = [para]
                    current_len = para_len
                else:
                    current_paras.append(para)
                    current_len += para_len

            if current_paras:
                combined_text = "\n\n".join(current_paras)
                if section_title and not combined_text.startswith(section_title):
                    chunk_text = f"[{section_title}]\n{combined_text}"
                else:
                    chunk_text = combined_text

                chunk_id = self.make_chunk_id(
                    document_id, f"struct_b{block_idx}_s{sub_idx}"
                )
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        text=chunk_text,
                        chunk_type="section" if section_title else (block.block_type or "text"),
                        page=block.page,
                        section=section_title or None,
                        slide=block.slide,
                        sheet=block.sheet,
                        source_locator=block.source_locator or f"section:{section_title or block_idx}",
                        token_count=self.estimate_tokens(chunk_text),
                        metadata={
                            **block.metadata,
                            "section_title": section_title,
                        },
                    )
                )

        return chunks
