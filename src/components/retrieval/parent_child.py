from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from .dense import DenseRetriever
from ..indexing.document_store import DocumentStore
from ..indexing.models import SearchResult

logger = logging.getLogger(__name__)


class ParentChildResult(BaseModel):
    """Retrieval result with both precise child reference and expanded parent context."""
    chunk_id: str
    document_id: str
    child_text: str
    parent_text: str
    score: float
    rank: int
    parent_id: Optional[str] = None
    source_locator: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    expanded: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def context_text(self) -> str:
        """Returns the expanded parent text if available, otherwise the child text."""
        return self.parent_text if self.expanded and self.parent_text else self.child_text


class ParentChildRetrievalResponse(BaseModel):
    """Structured response from parent-child retriever."""
    query: str
    candidate_chunks: int
    returned_chunks: int
    candidate_document_ids: List[str] = Field(default_factory=list)
    results: List[ParentChildResult] = Field(default_factory=list)


class ParentChildRetriever:
    """
    Two-stage retriever that queries fine-grained child chunks
    and resolves parent context using DocumentStore for comprehensive answer generation.
    """

    def __init__(
        self,
        base_retriever: Optional[DenseRetriever] = None,
        document_store: Optional[DocumentStore] = None,
        default_top_k: int = 5,
    ) -> None:
        self.base_retriever = base_retriever or DenseRetriever()
        self.doc_store = document_store or DocumentStore()
        self.default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        *,
        candidate_document_ids: Optional[Sequence[str]] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        expand_parent: bool = True,
    ) -> ParentChildRetrievalResponse:
        k = top_k or self.default_top_k
        dense_resp = self.base_retriever.retrieve(
            query=query,
            candidate_document_ids=candidate_document_ids,
            filter_criteria=filter_criteria,
            top_k=k,
        )

        results: List[ParentChildResult] = []
        for item in dense_resp.results:
            parent_id = item.parent_id or item.metadata.get("parent_chunk_id")
            parent_text = ""
            expanded = False

            if expand_parent and parent_id:
                stored = self.doc_store.get_parent_text(parent_id)
                if stored:
                    parent_text = stored
                    expanded = True

            results.append(
                ParentChildResult(
                    chunk_id=item.chunk_id,
                    document_id=item.document_id,
                    child_text=item.text,
                    parent_text=parent_text or item.text,
                    score=item.score,
                    rank=item.rank,
                    parent_id=parent_id,
                    source_locator=item.source_locator,
                    page=item.page,
                    section=item.section,
                    expanded=expanded,
                    metadata=item.metadata,
                )
            )

        return ParentChildRetrievalResponse(
            query=query,
            candidate_chunks=len(results),
            returned_chunks=len(results),
            candidate_document_ids=list(candidate_document_ids or []),
            results=results,
        )
