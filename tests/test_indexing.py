from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from src.components.chunking.models import Chunk
from src.components.indexing import (
    BM25Indexer,
    DocumentStore,
    IndexingPipeline,
    MockEmbedder,
    QdrantIndexer,
    SentenceTransformerEmbedder,
)
from src.components.ingestion.models import Document, DocumentBlock


def test_mock_embedder():
    embedder = MockEmbedder(dim=128)
    assert embedder.dimension == 128
    vecs = embedder.embed_texts(["hello world", "test query"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 128
    assert len(vecs[1]) == 128


def test_qdrant_indexer_lifecycle():
    qdrant = QdrantIndexer(collection_name="test_collection", vector_size=64)
    assert qdrant.health_check() is True

    chunk1 = Chunk(
        chunk_id="chk_1",
        document_id="doc_a",
        text="Vector search in high dimensions.",
        page=1,
        source_locator="page:1",
    )
    chunk2 = Chunk(
        chunk_id="chk_2",
        document_id="doc_b",
        text="Lexical retrieval using BM25.",
        page=2,
        source_locator="page:2",
    )
    emb = MockEmbedder(dim=64)
    vectors = emb.embed_texts([chunk1.text, chunk2.text])

    # Upsert
    upserted = qdrant.upsert_chunks([chunk1, chunk2], vectors)
    assert upserted == 2
    assert qdrant.count() == 2

    # Scoped search
    query_vec = emb.embed_query("vector search")
    hits_a = qdrant.search(query_vec, top_k=5, document_ids=["doc_a"])
    assert len(hits_a) == 1
    assert hits_a[0].chunk_id == "chk_1"

    # Delete
    deleted = qdrant.delete_by_document_id("doc_a")
    assert deleted is True


def test_bm25_indexer_persistence(tmp_path: Path):
    chunks_path = tmp_path / "chunks.json"
    bm25 = BM25Indexer(chunks_path=chunks_path)

    chunk = Chunk(
        chunk_id="chk_k1",
        document_id="doc_k",
        text="Attention mechanisms in deep learning models.",
        source_locator="doc:k",
    )
    bm25.upsert_chunks([chunk])
    assert bm25.count() == 1

    hits = bm25.search("attention", top_k=1)
    assert len(hits) == 1
    assert hits[0].chunk_id == "chk_k1"

    # Reload from disk
    bm25_reloaded = BM25Indexer(chunks_path=chunks_path)
    assert bm25_reloaded.count() == 1


def test_document_store_deduplication(tmp_path: Path):
    store_path = tmp_path / "store.json"
    store = DocumentStore(store_path=store_path)

    store.register_document("doc1", "paper.pdf", "sha_unique_1")
    store.update_status("doc1", "INDEXED", chunk_count=5)

    assert store.get_by_sha256("sha_unique_1") is not None
    assert store.get_by_sha256("sha_unknown") is None
    assert store.get("doc1").status == "INDEXED"


def test_indexing_pipeline_dedup(tmp_path: Path):
    embedder = MockEmbedder(dim=32)
    qdrant = QdrantIndexer(collection_name="test_pipe", vector_size=32)
    bm25 = BM25Indexer(chunks_path=tmp_path / "chunks.json")
    store = DocumentStore(store_path=tmp_path / "store.json")

    pipeline = IndexingPipeline(
        embedder=embedder,
        qdrant_indexer=qdrant,
        bm25_indexer=bm25,
        document_store=store,
    )

    doc = Document(
        document_id="d100",
        source_path="file.txt",
        filename="file.txt",
        extension=".txt",
        mime_type="text/plain",
        sha256="same_sha_123",
        size_bytes=100,
        blocks=[DocumentBlock(block_id="b1", text="Content to index.")],
    )

    # First index
    stats1 = pipeline.index_documents([doc])
    assert stats1.documents_indexed == 1

    # Second index (should skip due to SHA-256 deduplication)
    stats2 = pipeline.index_documents([doc])
    assert stats2.documents_indexed == 0
    assert stats2.documents_skipped == 1
