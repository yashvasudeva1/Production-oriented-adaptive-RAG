from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from ..indexing.embedder import BaseEmbedder, SentenceTransformerEmbedder
from ..indexing.models import SearchResult
from ..indexing.qdrant_indexer import QdrantIndexer

logger = logging.getLogger(__name__)


class DenseResult(BaseModel):
    """Ranked dense retrieval candidate."""
    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int
    parent_id: Optional[str] = None
    source_locator: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DenseRetrievalResponse(BaseModel):
    """Structured response from dense retriever."""
    query: str
    candidate_chunks: int
    returned_chunks: int
    candidate_document_ids: List[str] = Field(default_factory=list)
    results: List[DenseResult] = Field(default_factory=list)


class DenseRetriever:
    """
    Production dense retriever using Qdrant vector index and embeddings.
    Supports candidate document scoping, metadata filtering, and custom top-k.
    """

    def __init__(
        self,
        indexer: Optional[QdrantIndexer] = None,
        embedder: Optional[BaseEmbedder] = None,
        default_top_k: int = 10,
    ) -> None:
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.indexer = indexer or QdrantIndexer(vector_size=self.embedder.dimension)
        self.default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        *,
        candidate_document_ids: Optional[Sequence[str]] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> DenseRetrievalResponse:
        query = str(query or "").strip()
        k = top_k if top_k is not None else self.default_top_k
        k = max(1, min(k, 100))

        if not query:
            return DenseRetrievalResponse(
                query="",
                candidate_chunks=0,
                returned_chunks=0,
                candidate_document_ids=list(candidate_document_ids or []),
                results=[],
            )

        query_vector = self.embedder.embed_query(query)
        search_hits = self.indexer.search(
            query_vector=query_vector,
            top_k=k,
            document_ids=candidate_document_ids,
            filter_criteria=filter_criteria,
        )

        results: List[DenseResult] = []
        for hit in search_hits:
            results.append(
                DenseResult(
                    chunk_id=hit.chunk_id,
                    document_id=hit.document_id,
                    text=hit.text,
                    score=hit.score,
                    rank=hit.rank,
                    parent_id=hit.parent_id,
                    source_locator=hit.source_locator,
                    page=hit.page,
                    section=hit.section,
                    metadata=hit.metadata,
                )
            )

        return DenseRetrievalResponse(
            query=query,
            candidate_chunks=len(search_hits),
            returned_chunks=len(results),
            candidate_document_ids=list(candidate_document_ids or []),
            results=results,
        )
