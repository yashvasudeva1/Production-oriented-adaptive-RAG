from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Any, Dict, List, Optional, Sequence

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
    produces dense embeddings, and synchronizes Qdrant, BM25, and DocumentStore
    with two-phase transactional rollback, version tracking, and consistency checks.
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
        self.index_version = 1

    def index_documents(
        self,
        documents: Sequence[Document],
        force_reindex: bool = False,
    ) -> IndexingStats:
        """Process and index a batch of documents idempotently with rollback on failure."""
        start_time = time.perf_counter()
        stats = IndexingStats()

        for doc in documents:
            qdrant_upserted = False
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

                # Prepend a document-level summary chunk for improved summarization recall.
                try:
                    seen_texts = set()
                    summary_parts = []
                    for c in chunks:
                        key = c.text[:80]
                        if key not in seen_texts and len(summary_parts) < 3:
                            seen_texts.add(key)
                            summary_parts.append(c.text[:200])
                    if summary_parts:
                        from ..chunking.models import Chunk
                        summary_text = " [...] ".join(summary_parts)
                        summary_id = hashlib.md5(
                            f"summary:{doc.document_id}".encode()
                        ).hexdigest()[:16]
                        summary_chunk = Chunk(
                            chunk_id=f"summary_{summary_id}",
                            document_id=doc.document_id,
                            text=summary_text,
                            chunk_type="summary",
                            parent_id=None,
                            source_locator=chunks[0].source_locator if chunks else None,
                            page=None,
                            section="Document Overview",
                            metadata={**doc.metadata, "is_summary": True},
                        )
                        chunks = [summary_chunk] + list(chunks)
                except Exception as _sum_exc:
                    logger.debug(f"Summary chunk generation skipped: {_sum_exc}")

                # Cache raw parent chunk text so parent-child retrieval can expand without re-reading files
                for c in chunks:
                    if c.chunk_type == "parent":
                        self.doc_store.store_parent_text(c.chunk_id, c.text)

                # Ingest-time chunk deduplication across documents
                existing_hashes = {
                    hashlib.md5(re.sub(r"\s+", " ", c.get("text", "")).strip().lower().encode()).hexdigest()
                    for c in getattr(self.bm25, "_all_chunks", [])
                    if c.get("document_id") != doc.document_id
                }
                deduped_chunks = []
                for c in chunks:
                    c_hash = hashlib.md5(re.sub(r"\s+", " ", c.text).strip().lower().encode()).hexdigest()
                    if c_hash in existing_hashes:
                        logger.info(f"Skipping cross-document duplicate chunk {c.chunk_id} from {doc.filename}")
                        continue
                    existing_hashes.add(c_hash)
                    deduped_chunks.append(c)
                chunks = deduped_chunks

                chunk_texts = [c.text for c in chunks]
                embeddings = self.embedder.embed_texts(chunk_texts)

                # Cache vectors to disk to enable instant sub-second app restarts
                try:
                    import numpy as np
                    from pathlib import Path
                    dim = getattr(self.embedder, "dimension", len(embeddings[0]) if embeddings else 384)
                    cache_file = Path(__file__).resolve().parents[3] / "metadata" / f"chunk_embeddings_{dim}.npz"
                    cache_dict = {}
                    if cache_file.exists():
                        try:
                            c_data = np.load(cache_file)
                            cache_dict = {k: c_data[k] for k in c_data.files}
                        except Exception:
                            pass
                    for c, v in zip(chunks, embeddings):
                        cache_dict[c.chunk_id] = np.array(v, dtype=np.float32)
                    np.savez_compressed(cache_file, **cache_dict)
                except Exception as _c_exc:
                    logger.debug(f"Vector disk caching skipped: {_c_exc}")

                # Two-phase transaction: Qdrant first, then BM25 + DocumentStore
                self.qdrant.upsert_chunks(chunks, embeddings)
                qdrant_upserted = True

                self.bm25.upsert_chunks(chunks)

                self.doc_store.update_status(
                    doc.document_id, status="INDEXED", chunk_count=len(chunks)
                )

                stats.documents_indexed += 1
                stats.chunks_indexed += len(chunks)

            except Exception as exc:
                logger.error(f"Failed to index document {doc.filename}: {exc}")
                # Rollback partial Qdrant writes if failure occurred downstream
                if qdrant_upserted:
                    try:
                        logger.warning(f"Rolling back Qdrant points for failed document {doc.document_id}")
                        self.qdrant.delete_by_document_id(doc.document_id)
                    except Exception as rb_exc:
                        logger.error(f"Failed to rollback Qdrant: {rb_exc}")
                self.doc_store.update_status(
                    doc.document_id, status="FAILED", error_message=str(exc)
                )
                stats.errors.append({"document_id": doc.document_id, "error": str(exc)})

        if stats.documents_indexed > 0:
            self.index_version += 1
            from ..cache import CacheManager
            CacheManager.get_instance().bump_index_version()

        stats.duration_ms = (time.perf_counter() - start_time) * 1000
        return stats

    def index_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Directly index a pre-generated sequence of Chunks with failure recovery."""
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)
        qdrant_done = False
        try:
            self.qdrant.upsert_chunks(chunks, embeddings)
            qdrant_done = True
            self.bm25.upsert_chunks(chunks)
            self.index_version += 1
            from ..cache import CacheManager
            CacheManager.get_instance().bump_index_version()
            return len(chunks)
        except Exception as exc:
            logger.error(f"Failed to index chunks: {exc}")
            if qdrant_done:
                try:
                    c_ids = [c.chunk_id for c in chunks]
                    self.qdrant.client.delete(
                        collection_name=self.qdrant.collection_name,
                        points_selector=c_ids,
                    )
                except Exception:
                    pass
            raise

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks consistently from Qdrant, BM25, and DocumentStore."""
        q_deleted = self.qdrant.delete_by_document_id(document_id)
        b_deleted = self.bm25.delete_by_document_id(document_id)
        s_deleted = self.doc_store.delete(document_id)
        self.index_version += 1
        from ..cache import CacheManager
        CacheManager.get_instance().bump_index_version()
        return q_deleted or b_deleted or s_deleted

    def verify_index_consistency(self) -> Dict[str, Any]:
        """
        Verify synchronization and consistency across Qdrant, BM25, and DocumentStore.
        Reports any missing or orphan documents/chunks.
        """
        doc_store_docs = {d.document_id: d for d in self.doc_store.list_documents() if d.status == "INDEXED"}
        bm25_chunks = getattr(self.bm25, "_all_chunks", [])
        bm25_doc_ids = {c.get("document_id") for c in bm25_chunks if c.get("document_id")}

        qdrant_points = self.qdrant.count()
        bm25_chunk_count = len(bm25_chunks)

        discrepancies: List[str] = []
        orphaned_in_bm25 = set()
        orphaned_in_qdrant = set()

        if set(doc_store_docs.keys()) != bm25_doc_ids:
            missing_in_bm25 = set(doc_store_docs.keys()) - bm25_doc_ids
            orphaned_in_bm25 = bm25_doc_ids - set(doc_store_docs.keys())
            if missing_in_bm25:
                discrepancies.append(f"Documents in store but missing from BM25: {missing_in_bm25}")
            if orphaned_in_bm25:
                discrepancies.append(f"Documents in BM25 but missing from store: {orphaned_in_bm25}")

        if qdrant_points != bm25_chunk_count and qdrant_points > 0 and bm25_chunk_count > 0:
            discrepancies.append(
                f"Chunk count mismatch: Qdrant has {qdrant_points} points, BM25 has {bm25_chunk_count} chunks"
            )

        is_consistent = len(discrepancies) == 0
        return {
            "consistent": is_consistent,
            "is_consistent": is_consistent,
            "index_version": self.index_version,
            "document_store_count": len(doc_store_docs),
            "bm25_chunk_count": bm25_chunk_count,
            "qdrant_point_count": qdrant_points,
            "discrepancies": discrepancies,
            "orphaned_in_bm25": list(orphaned_in_bm25),
        }

    def repair_index_inconsistencies(self) -> Dict[str, Any]:
        """Repair index discrepancies by resynchronizing BM25 chunks into Qdrant or pruning orphans."""
        check = self.verify_index_consistency()
        if check["consistent"]:
            return {"status": "already_consistent", "repaired": True, "repaired_count": 0}

        repaired_count = 0
        bm25_chunks = getattr(self.bm25, "_all_chunks", [])
        
        # 1. Resynchronize Qdrant from BM25 if Qdrant is missing points
        if len(bm25_chunks) > 0 and self.qdrant.count() == 0:
            logger.info("Resynchronizing Qdrant collection from BM25 corpus...")
            loaded = [Chunk.from_dict(c) for c in bm25_chunks]
            vecs = self.embedder.embed_texts([c.text for c in loaded])
            self.qdrant.upsert_chunks(loaded, vecs)
            repaired_count += len(loaded)

        # 2. Prune BM25 chunks that don't exist in doc_store
        doc_store_docs = {d.document_id for d in self.doc_store.list_documents() if d.status == "INDEXED"}
        if check.get("orphaned_in_bm25"):
            for orphan_id in check["orphaned_in_bm25"]:
                self.bm25.delete_by_document_id(orphan_id)
                self.qdrant.delete_by_document_id(orphan_id)
                repaired_count += 1

        self.index_version += 1
        from ..cache import CacheManager
        CacheManager.get_instance().bump_index_version()

        post_check = self.verify_index_consistency()
        return {
            "status": "repaired",
            "repaired": True,
            "chunks_re-indexed": repaired_count,
            "post_check": post_check,
        }
