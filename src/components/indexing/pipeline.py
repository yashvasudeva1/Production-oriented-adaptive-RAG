from __future__ import annotations

import logging
import time
from typing import List, Optional, Sequence

from ..chunking.models import Chunk
from ..chunking.pipeline import ChunkingPipeline
from ..ingestion.models import Document
from .bm25_indexer import BM25Indexer
from .document_store import DocumentStore
from .embedder import BaseEmbedder, SentenceTransformerEmbedder
from .models import IndexingStats
from .qdrant_indexer import QdrantIndexer

logger = logging.getLogger(__name__)


class IndexingPipeline:
    """
    Unified indexing pipeline that consumes Documents or Chunks,
    produces dense embeddings, and synchronizes Qdrant, BM25, and DocumentStore.
    """

    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        qdrant_indexer: Optional[QdrantIndexer] = None,
        bm25_indexer: Optional[BM25Indexer] = None,
        document_store: Optional[DocumentStore] = None,
        chunking_pipeline: Optional[ChunkingPipeline] = None,
    ) -> None:
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.qdrant = qdrant_indexer or QdrantIndexer(vector_size=self.embedder.dimension)
        self.bm25 = bm25_indexer or BM25Indexer()
        self.doc_store = document_store or DocumentStore()
        self.chunker = chunking_pipeline or ChunkingPipeline()

    def index_documents(
        self,
        documents: Sequence[Document],
        force_reindex: bool = False,
    ) -> IndexingStats:
        """Process and index a batch of documents idempotently."""
        start_time = time.perf_counter()
        stats = IndexingStats()

        for doc in documents:
            try:
                # Skip re-indexing if content hash has already been processed
                existing = self.doc_store.get_by_sha256(doc.sha256)
                if existing and not force_reindex:
                    logger.info(
                        f"Skipping document {doc.filename} (already indexed as {existing.document_id})"
                    )
                    stats.documents_skipped += 1
                    continue

                self.doc_store.register_document(
                    document_id=doc.document_id,
                    filename=doc.filename,
                    sha256=doc.sha256,
                    size_bytes=doc.size_bytes,
                    metadata=doc.metadata,
                    status="INGESTING",
                )

                chunks = self.chunker.chunk_document(doc)
                if not chunks:
                    self.doc_store.update_status(
                        doc.document_id, status="FAILED", error_message="No text chunks generated"
                    )
                    continue

                # Cache raw parent chunk text so parent-child retrieval can expand without re-reading files
                for c in chunks:
                    if c.chunk_type == "parent":
                        self.doc_store.store_parent_text(c.chunk_id, c.text)

                chunk_texts = [c.text for c in chunks]
                embeddings = self.embedder.embed_texts(chunk_texts)

                # Keep vector index and keyword index synchronized
                self.qdrant.upsert_chunks(chunks, embeddings)
                self.bm25.upsert_chunks(chunks)

                self.doc_store.update_status(
                    doc.document_id, status="INDEXED", chunk_count=len(chunks)
                )

                stats.documents_indexed += 1
                stats.chunks_indexed += len(chunks)

            except Exception as exc:
                logger.error(f"Failed to index document {doc.filename}: {exc}")
                self.doc_store.update_status(
                    doc.document_id, status="FAILED", error_message=str(exc)
                )
                stats.errors.append({"document_id": doc.document_id, "error": str(exc)})

        stats.duration_ms = (time.perf_counter() - start_time) * 1000
        return stats

    def index_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Directly index a pre-generated sequence of Chunks."""
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)
        self.qdrant.upsert_chunks(chunks, embeddings)
        self.bm25.upsert_chunks(chunks)
        return len(chunks)

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks from Qdrant, BM25, and DocumentStore."""
        q_deleted = self.qdrant.delete_by_document_id(document_id)
        b_deleted = self.bm25.delete_by_document_id(document_id)
        s_deleted = self.doc_store.delete(document_id)
        return q_deleted or b_deleted or s_deleted
