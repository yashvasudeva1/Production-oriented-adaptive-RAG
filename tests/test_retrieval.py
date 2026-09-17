from __future__ import annotations

from pathlib import Path
import pytest

from src.components.chunking.models import Chunk
from src.components.dense_retrieval import DenseRetriever
from src.components.indexing import (
    BM25Indexer,
    DocumentStore,
    MockEmbedder,
    QdrantIndexer,
    SearchResult,
)
from src.components.multi_query_retrieval import MultiQueryRetriever
from src.components.parent_child_retrieval import ParentChildRetriever
from src.components.retrieval.fusion import reciprocal_rank_fusion


def test_dense_retrieval_scoping():
    emb = MockEmbedder(dim=32)
    qdrant = QdrantIndexer(collection_name="test_dense_ret", vector_size=32)
    retriever = DenseRetriever(indexer=qdrant, embedder=emb)

    chunk_a = Chunk(chunk_id="ca", document_id="doc_A", text="Quantum computing algorithms.")
    chunk_b = Chunk(chunk_id="cb", document_id="doc_B", text="Classical sorting algorithms.")
    vecs = emb.embed_texts([chunk_a.text, chunk_b.text])
    qdrant.upsert_chunks([chunk_a, chunk_b], vecs)

    # Search with scope to doc_A only
    resp = retriever.retrieve("computing", candidate_document_ids=["doc_A"])
    assert len(resp.results) == 1
    assert resp.results[0].chunk_id == "ca"


def test_parent_child_retriever(tmp_path: Path):
    emb = MockEmbedder(dim=32)
    qdrant = QdrantIndexer(collection_name="test_pc_ret", vector_size=32)
    doc_store = DocumentStore(store_path=tmp_path / "pc_store.json")

    parent_id = "parent_001"
    parent_text = "Full section: Reinforcement learning from human feedback (RLHF) optimizes models."
    doc_store.store_parent_text(parent_id, parent_text)

    child_chunk = Chunk(
        chunk_id="child_001",
        document_id="doc_rlhf",
        parent_id=parent_id,
        text="RLHF optimizes models.",
        chunk_type="child",
    )
    vecs = emb.embed_texts([child_chunk.text])
    qdrant.upsert_chunks([child_chunk], vecs)

    pc_retriever = ParentChildRetriever(
        base_retriever=DenseRetriever(indexer=qdrant, embedder=emb),
        document_store=doc_store,
    )

    resp = pc_retriever.retrieve("RLHF", expand_parent=True)
    assert len(resp.results) == 1
    assert resp.results[0].expanded is True
    assert resp.results[0].context_text == parent_text


def test_multi_query_retriever():
    emb = MockEmbedder(dim=32)
    qdrant = QdrantIndexer(collection_name="test_mq_ret", vector_size=32)
    chunk = Chunk(chunk_id="cmq", document_id="doc_m", text="Microservices architecture design.")
    qdrant.upsert_chunks([chunk], emb.embed_texts([chunk.text]))

    mq = MultiQueryRetriever(
        base_retriever=DenseRetriever(indexer=qdrant, embedder=emb),
        num_queries=2,
    )
    resp = mq.retrieve("microservices")
    assert len(resp.generated_queries) >= 2
    assert len(resp.results) >= 1
    assert resp.results[0].chunk_id == "cmq"


def test_reciprocal_rank_fusion():
    list1 = [
        SearchResult(chunk_id="c1", document_id="d1", text="text1", score=0.9, rank=1, retriever_name="dense"),
        SearchResult(chunk_id="c2", document_id="d1", text="text2", score=0.8, rank=2, retriever_name="dense"),
    ]
    list2 = [
        SearchResult(chunk_id="c2", document_id="d1", text="text2", score=10.0, rank=1, retriever_name="bm25"),
        SearchResult(chunk_id="c3", document_id="d2", text="text3", score=5.0, rank=2, retriever_name="bm25"),
    ]

    fused = reciprocal_rank_fusion([list1, list2], k=60, top_n=5)
    assert len(fused) == 3
    # c2 appeared in both lists and should have highest combined RRF score
    assert fused[0].chunk_id == "c2"
    assert "dense" in fused[0].provenance_sources
    assert "bm25" in fused[0].provenance_sources
