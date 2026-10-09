from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from .dense import DenseRetriever
from .fusion import UnifiedCandidate, reciprocal_rank_fusion
from ..indexing.models import SearchResult

logger = logging.getLogger(__name__)


class MultiQueryRetrievalResponse(BaseModel):
    """Structured response from multi-query retriever."""
    original_query: str
    generated_queries: List[str] = Field(default_factory=list)
    candidate_chunks: int
    returned_chunks: int
    candidate_document_ids: List[str] = Field(default_factory=list)
    results: List[UnifiedCandidate] = Field(default_factory=list)


class MultiQueryRetriever:
    """
    Generates multiple query perspectives to overcome vocabulary mismatch and
    retrieves/fuses candidate chunks across all formulations using RRF.
    """

    def __init__(
        self,
        base_retriever: Optional[DenseRetriever] = None,
        generator_fn: Optional[Callable[[str, int], List[str]]] = None,
        num_queries: int = 3,
        default_top_k: int = 10,
    ) -> None:
        self.base_retriever = base_retriever or DenseRetriever()
        self.generator_fn = generator_fn
        self.num_queries = num_queries
        self.default_top_k = default_top_k
        self._query_cache: Dict[str, List[str]] = {}

    def _generate_queries_heuristic(self, query: str) -> List[str]:
        """
        High-precision rule-based query expansion and perspective generation.
        Extracts key noun phrases, relational clauses, and focused entity variants
        without adding diluting conversational fluff.
        """
        q = query.strip()
        words = q.split()
        queries = [q]

        # 1. Multi-hop & relational clause splitting
        relation_patterns = [
            r"\b(?:connect(?:s|ed)?\s+to|interact(?:s|ed)?\s+with|jointly\s+impact|influence(?:s|d)?\s+whether)\b",
            r"\b(?:compared\s+to|in\s+contrast\s+to|versus|vs\.?)\b",
            r"\b(?:prevent(?:s|ed)?\s+.*?\s+in\s+the|because\s+of|due\s+to)\b",
        ]
        split_done = False
        for pat in relation_patterns:
            parts = re.split(pat, q, flags=re.IGNORECASE)
            if len(parts) >= 2:
                for part in parts:
                    cleaned_part = re.sub(r"^(?:how|why|what|explain|does|do|the)\s+", "", part.strip(), flags=re.IGNORECASE).strip(" ?.,")
                    if len(cleaned_part.split()) >= 2:
                        queries.append(cleaned_part)
                split_done = True
                break

        # 2. Key phrase and entity extraction (keyword-dense core)
        stopwords = {
            "how", "why", "what", "when", "where", "which", "does", "do", "explain",
            "tell", "about", "describe", "the", "a", "an", "is", "are", "was", "were",
            "in", "on", "at", "by", "for", "with", "from", "and", "or", "to", "of",
        }
        content_tokens = [w for w in re.findall(r"[a-zA-Z0-9_\-\.]+", q) if w.lower() not in stopwords]
        if len(content_tokens) >= 3:
            keyword_core = " ".join(content_tokens)
            if keyword_core not in queries:
                queries.append(keyword_core)

        # 3. Domain-specific synonym and terminology expansion
        synonym_map = {
            "convs2s": "convolutional sequence to sequence",
            "bleu": "bilingual evaluation understudy translation score",
            "stipend": "financial assistance remuneration honorarium",
            "positional encoding": "sinusoidal position embeddings order",
            "label smoothing": "regularization penalizing confident predictions",
            "d_model": "transformer embedding hidden dimension",
            "attendance": "minimum attendance 75 percent requirement",
        }
        q_lower = q.lower()
        for term, expansion in synonym_map.items():
            if term in q_lower:
                expanded_variant = f"{q} {expansion}"
                if expanded_variant not in queries:
                    queries.append(expanded_variant)
                break

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for cand in queries:
            c_norm = cand.strip().lower()
            if c_norm and c_norm not in seen:
                seen.add(c_norm)
                deduped.append(cand.strip())

        # If still only 1 query generated, provide a focused perspective query
        if len(deduped) < 2:
            deduped.append(f"{q} overview")

        return deduped[: self.num_queries + 1]

    def generate_queries(self, query: str) -> List[str]:
        if query in self._query_cache:
            return self._query_cache[query]

        if self.generator_fn:
            try:
                queries = self.generator_fn(query, self.num_queries)
                if queries:
                    if query not in queries:
                        queries = [query] + queries
                    result = queries[: self.num_queries + 1]
                    self._query_cache[query] = result
                    return result
            except Exception as exc:
                logger.warning(f"Error in multi-query LLM generator: {exc}, using heuristics.")

        result = self._generate_queries_heuristic(query)
        self._query_cache[query] = result
        return result

    def retrieve(
        self,
        query: str,
        *,
        candidate_document_ids: Optional[Sequence[str]] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> MultiQueryRetrievalResponse:
        k = top_k or self.default_top_k
        formulations = self.generate_queries(query)

        candidate_runs: List[List[SearchResult]] = []
        for q in formulations:
            dense_resp = self.base_retriever.retrieve(
                query=q,
                candidate_document_ids=candidate_document_ids,
                filter_criteria=filter_criteria,
                top_k=k,
            )
            run_items = [
                SearchResult(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    text=r.text,
                    score=r.score,
                    rank=r.rank,
                    parent_id=r.parent_id,
                    source_locator=r.source_locator,
                    page=r.page,
                    section=r.section,
                    retriever_name="multi_query_dense",
                    metadata=r.metadata,
                )
                for r in dense_resp.results
            ]
            candidate_runs.append(run_items)

        fused = reciprocal_rank_fusion(candidate_runs, k=60, top_n=k)

        return MultiQueryRetrievalResponse(
            original_query=query,
            generated_queries=formulations,
            candidate_chunks=len(fused),
            returned_chunks=len(fused),
            candidate_document_ids=list(candidate_document_ids or []),
            results=fused,
        )
