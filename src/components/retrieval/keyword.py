from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from ..indexing.bm25_indexer import BM25Indexer
from ..indexing.models import SearchResult

logger = logging.getLogger(__name__)


class KeywordRetrievalResponse(BaseModel):
    """Structured response from keyword retriever."""
    query: str
    candidate_chunks: int
    returned_chunks: int
    candidate_document_ids: List[str] = Field(default_factory=list)
    results: List[SearchResult] = Field(default_factory=list)


class KeywordRetriever:
    """
    Production lexical keyword retriever using BM25 index.
    Supports candidate document scoping, score normalization, and custom top-k.
    """

    def __init__(
        self,
        indexer: Optional[BM25Indexer] = None,
        default_top_k: int = 10,
    ) -> None:
        self.indexer = indexer or BM25Indexer()
        self.default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        *,
        candidate_document_ids: Optional[Sequence[str]] = None,
        retrieval_signals: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> KeywordRetrievalResponse:
        query = str(query or "").strip()
        k = top_k if top_k is not None else self.default_top_k
        k = max(1, min(k, 100))

        if not query:
            return KeywordRetrievalResponse(
                query="",
                candidate_chunks=0,
                returned_chunks=0,
                candidate_document_ids=list(candidate_document_ids or []),
                results=[],
            )

        results = self.indexer.search(
            query=query,
            top_k=k,
            candidate_document_ids=candidate_document_ids,
            retrieval_signals=retrieval_signals,
        )

        return KeywordRetrievalResponse(
            query=query,
            candidate_chunks=len(results),
            returned_chunks=len(results),
            candidate_document_ids=list(candidate_document_ids or []),
            results=results,
        )
