from __future__ import annotations

from typing import Dict, List, Sequence

from ..ingestion.models import DocumentBlock
from .base import BaseChunker
from .code import CodeChunker
from .config import ChunkingConfig
from .markdown import MarkdownChunker
from .models import Chunk
from .parent_child import ParentChildChunker
from .recursive import RecursiveChunker
from .semantic import SemanticChunker
from .structural import StructuralChunker
from .table import TableChunker
from .transcript import TranscriptChunker


class ChunkRouter:
    """
    Intelligent router that selects specialized chunkers based on
    block type and configured strategy.
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()
        self._recursive = RecursiveChunker(self.config)
        self._structural = StructuralChunker(self.config)
        self._markdown = MarkdownChunker(self.config)
        self._code = CodeChunker(self.config)
        self._table = TableChunker(self.config)
        self._transcript = TranscriptChunker(self.config)
        self._parent_child = ParentChildChunker(self.config)
        self._semantic = SemanticChunker(self.config)

    def route_and_chunk(
        self, blocks: Sequence[DocumentBlock], document_id: str
    ) -> List[Chunk]:
        strategy = self.config.strategy

        # If user explicitly requested a non-auto global strategy:
        if strategy == "recursive":
            return self._recursive.chunk(blocks, document_id)
        elif strategy == "structural":
            return self._structural.chunk(blocks, document_id)
        elif strategy == "markdown":
            return self._markdown.chunk(blocks, document_id)
        elif strategy == "code":
            return self._code.chunk(blocks, document_id)
        elif strategy == "table":
            return self._table.chunk(blocks, document_id)
        elif strategy == "transcript":
            return self._transcript.chunk(blocks, document_id)
        elif strategy == "parent_child":
            return self._parent_child.chunk(blocks, document_id)
        elif strategy == "semantic":
            return self._semantic.chunk(blocks, document_id)

        # "auto" strategy: group blocks by type and dispatch to specialized chunkers
        chunks: List[Chunk] = []
        for block in blocks:
            b_type = (block.block_type or "").lower()

            if b_type in ("code", "source_code", "python", "javascript", "sql"):
                chunks.extend(self._code.chunk([block], document_id))
            elif b_type in ("table", "spreadsheet", "csv", "tsv"):
                chunks.extend(self._table.chunk([block], document_id))
            elif b_type in ("markdown", "heading"):
                chunks.extend(self._markdown.chunk([block], document_id))
            elif b_type in ("transcript", "audio", "video", "utterance"):
                chunks.extend(self._transcript.chunk([block], document_id))
            elif block.section:
                chunks.extend(self._structural.chunk([block], document_id))
            else:
                chunks.extend(self._recursive.chunk([block], document_id))

        return chunks
