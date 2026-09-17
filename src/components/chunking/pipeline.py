from __future__ import annotations

import logging
from typing import List, Sequence

from ..ingestion.models import Document, DocumentBlock
from .config import ChunkingConfig
from .models import Chunk
from .router import ChunkRouter

logger = logging.getLogger(__name__)


class ChunkingPipeline:
    """
    Main entrypoint for chunking documents.
    Consumes Document or DocumentBlock[] and produces fully validated Chunk[].
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()
        self.router = ChunkRouter(self.config)

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Chunk a complete Document and preserve its document_id and metadata."""
        if not document.blocks:
            logger.warning(f"Document {document.document_id} has no blocks to chunk.")
            return []

        chunks = self.router.route_and_chunk(
            document.blocks, document_id=document.document_id
        )

        # Enrich chunk metadata with document-level context
        for chunk in chunks:
            chunk.metadata.setdefault("filename", document.filename)
            chunk.metadata.setdefault("extension", document.extension)
            chunk.metadata.setdefault("sha256", document.sha256)
            if document.metadata:
                chunk.metadata.setdefault("doc_metadata", document.metadata)

        self._validate_chunks(chunks)
        return chunks

    def chunk_blocks(
        self, blocks: Sequence[DocumentBlock], document_id: str
    ) -> List[Chunk]:
        """Chunk raw document blocks directly."""
        chunks = self.router.route_and_chunk(blocks, document_id=document_id)
        self._validate_chunks(chunks)
        return chunks

    def _validate_chunks(self, chunks: List[Chunk]) -> None:
        """Enforce strict provenance contracts on all generated chunks."""
        seen_ids = set()
        for i, chunk in enumerate(chunks):
            if not chunk.chunk_id:
                raise ValueError(f"Chunk at index {i} has empty chunk_id")
            if not chunk.document_id:
                raise ValueError(f"Chunk {chunk.chunk_id} has empty document_id")
            if not chunk.text or not chunk.text.strip():
                raise ValueError(f"Chunk {chunk.chunk_id} has empty text")
            if chunk.source_locator is None:
                raise ValueError(f"Chunk {chunk.chunk_id} missing source_locator")
            if chunk.chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk_id detected: {chunk.chunk_id}")
            seen_ids.add(chunk.chunk_id)
