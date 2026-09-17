from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from ..generation.generator import GroundedGenerator
from ..indexing.bm25_indexer import BM25Indexer
from ..indexing.document_store import DocumentStore
from ..indexing.embedder import BaseEmbedder, SentenceTransformerEmbedder
from ..indexing.models import SearchResult
from ..indexing.qdrant_indexer import QdrantIndexer
from ..query.models import QueryPlan
from ..query.planner import QueryPlanner
from ..reranker import CrossEncoderReranker, RerankerResult
from ..retrieval.confidence import RetrievalConfidence, RetrievalConfidenceScorer
from ..retrieval.dense import DenseRetriever
from ..retrieval.fusion import UnifiedCandidate, reciprocal_rank_fusion
from ..retrieval.hybrid import HybridRetriever
from ..retrieval.keyword import KeywordRetriever
from ..retrieval.multi_query import MultiQueryRetriever
from ..retrieval.parent_child import ParentChildRetriever
from .context_filter import ContextFilter
from .evidence_gate import EvidenceGate, EvidenceVerdict

logger = logging.getLogger(__name__)


class RetrievalMetadata(BaseModel):
    strategy: str
    query_type: str
    execution_mode: str = "balanced"
    cascade_stages_executed: List[str] = Field(default_factory=list)
    escalated: bool = False
    escalation_reason: Optional[str] = None
    reranked: bool = False
    parent_expanded: bool = False
    multi_query_used: bool = False
    decomposed: bool = False
    fast_rank: Optional[int] = None
    balanced_rank: Optional[int] = None
    deep_rank: Optional[int] = None
    final_rank: Optional[int] = None
    dense_candidates: int = 0
    keyword_candidates: int = 0
    multi_query_candidates: int = 0
    fused_candidates: int = 0
    reranked_candidates: int = 0
    filtered_context_candidates: int = 0
    latency_breakdown_ms: Dict[str, float] = Field(default_factory=dict)


class RAGResponse(BaseModel):
    """Complete, structured response from the ResearchLens system."""
    query: str
    answer: str
    confidence: float
    abstained: bool = False
    abstention_reason: Optional[str] = None
    query_type: str = "general"
    execution_mode: str = "balanced"
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval: RetrievalMetadata
    citation_validation: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0


class RAGOrchestrator:
    """
    Quality-Preserving Additive Confidence Cascade Orchestrator.
    Core Philosophy: CHEAP FIRST -> MEASURE CONFIDENCE -> ESCALATE ADDITIVELY ONLY IF NECESSARY.

    Monotonic Invariant:
    All candidates retrieved in earlier stages are retained and merged into later stages
    via additive reciprocal rank fusion with exact-match protection.
    """

    def __init__(
        self,
        query_planner: Optional[QueryPlanner] = None,
        embedder: Optional[BaseEmbedder] = None,
        qdrant_indexer: Optional[QdrantIndexer] = None,
        bm25_indexer: Optional[BM25Indexer] = None,
        document_store: Optional[DocumentStore] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        context_filter: Optional[ContextFilter] = None,
        evidence_gate: Optional[EvidenceGate] = None,
        generator: Optional[GroundedGenerator] = None,
        confidence_scorer: Optional[RetrievalConfidenceScorer] = None,
    ) -> None:
        self.doc_store = document_store or DocumentStore()
        self.planner = query_planner or QueryPlanner(document_store=self.doc_store)
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.qdrant = qdrant_indexer or QdrantIndexer(vector_size=self.embedder.dimension)
        self.bm25 = bm25_indexer or BM25Indexer()

        self.dense_retriever = DenseRetriever(
            indexer=self.qdrant, embedder=self.embedder
        )
        self.keyword_retriever = KeywordRetriever(
            indexer=self.bm25
        )
        self.hybrid_retriever = HybridRetriever(
            dense_retriever=self.dense_retriever,
            keyword_retriever=self.keyword_retriever,
        )
        self.parent_child_retriever = ParentChildRetriever(
            base_retriever=self.dense_retriever, document_store=self.doc_store
        )
        self.multi_query_retriever = MultiQueryRetriever(
            base_retriever=self.dense_retriever
        )

        self.reranker = reranker or CrossEncoderReranker()
        self.context_filter = context_filter or ContextFilter()
        self.evidence_gate = evidence_gate or EvidenceGate()
        self.generator = generator or GroundedGenerator()
        self.confidence_scorer = confidence_scorer or RetrievalConfidenceScorer()

    def _search_keyword(
        self, query: str, candidate_document_ids: Optional[List[str]], top_k: int, signals: Dict[str, Any]
    ) -> List[SearchResult]:
        return self.bm25.search(
            query=query,
            candidate_document_ids=candidate_document_ids,
            retrieval_signals=signals,
            top_k=top_k,
        )

    def _search_dense(
        self, query: str, candidate_document_ids: Optional[List[str]], top_k: int
    ) -> List[SearchResult]:
        d_resp = self.dense_retriever.retrieve(
            query=query,
            candidate_document_ids=candidate_document_ids,
            top_k=top_k,
        )
        return [
            SearchResult(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                text=c.text,
                score=c.score,
                rank=c.rank,
                parent_id=c.parent_id,
                source_locator=c.source_locator,
                page=c.page,
                section=c.section,
                chunk_type="text",
                retriever_name="dense_qdrant",
                metadata=c.metadata,
            )
            for c in d_resp.results
        ]

    def query(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        mode: Optional[str] = None,
    ) -> RAGResponse:
        start_time = time.perf_counter()
        latencies: Dict[str, float] = {}

        # 1. Tiered Query Planning (Level 0 Normalization + Level 1 Signals)
        t_plan = time.perf_counter()
        plan = self.planner.plan(
            query,
            explicit_document_ids=document_ids,
            forced_mode=mode if mode in ("fast", "balanced", "deep") else None,
        )
        latencies["planning_ms"] = (time.perf_counter() - t_plan) * 1000

        c_doc_ids = plan.candidate_document_ids or document_ids
        active_mode = plan.execution_mode
        cascade_stages: List[str] = []
        escalation_reason: Optional[str] = None

        # Additive candidate accumulator across all stages
        all_candidate_runs: List[List[Any]] = []

        dense_count = 0
        keyword_count = 0
        mq_count = 0
        fused_count = 0
        reranked_count = 0

        did_rerank = False
        did_parent_expand = False
        did_multi_query = False
        did_decompose = False

        exact_ids = plan.retrieval_signals.get("exact_identifiers", [])

        final_context: List[RerankerResult] = []
        verdict: Optional[EvidenceVerdict] = None

        # -----------------------------------------------------------------
        # STAGE 1: FAST ROUTE (BM25 lexical matching in < 1ms)
        # -----------------------------------------------------------------
        if active_mode == "fast":
            cascade_stages.append("fast_bm25")
            t_ret = time.perf_counter()
            kw_hits = self._search_keyword(
                plan.normalized_query, c_doc_ids, plan.top_k_keyword, plan.retrieval_signals
            )
            latencies["fast_retrieval_ms"] = (time.perf_counter() - t_ret) * 1000
            keyword_count = len(kw_hits)
            all_candidate_runs.append(kw_hits)

            # Measure candidate confidence
            conf = self.confidence_scorer.score_candidates(
                query=plan.normalized_query,
                candidates=kw_hits,
                execution_mode="fast",
                exact_identifiers=exact_ids,
            )

            if conf.is_sufficient:
                bypass_results = self.reranker.bypass_rerank(kw_hits, top_k=plan.top_k)
                ctx = self.context_filter.filter_context(bypass_results, max_tokens=plan.context_budget)
                v_fast = self.evidence_gate.evaluate(plan.normalized_query, ctx)

                if v_fast.allowed:
                    final_context = ctx
                    verdict = v_fast
                    reranked_count = len(bypass_results)
                else:
                    if mode != "fast":
                        escalation_reason = f"Fast route rejected by evidence gate: {v_fast.reason}"
                        active_mode = "balanced"
                    else:
                        verdict = v_fast
            else:
                if mode != "fast":
                    escalation_reason = f"Fast BM25 confidence weak: {conf.reason}"
                    active_mode = "balanced"
                else:
                    bypass_results = self.reranker.bypass_rerank(kw_hits, top_k=plan.top_k)
                    final_context = self.context_filter.filter_context(bypass_results, max_tokens=plan.context_budget)
                    verdict = self.evidence_gate.evaluate(plan.normalized_query, final_context)

        # -----------------------------------------------------------------
        # STAGE 2: BALANCED ROUTE (Additive Hybrid: Dense + BM25 with RRF)
        # -----------------------------------------------------------------
        if active_mode == "balanced" and verdict is None:
            cascade_stages.append("balanced_hybrid")
            t_ret = time.perf_counter()

            # Execute dense retrieval concurrently if keyword already ran, or run both
            if not any("bm25" in getattr(r[0], "retriever_name", "") for r in all_candidate_runs if r):
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    fut_kw = executor.submit(
                        self._search_keyword, plan.normalized_query, c_doc_ids, plan.top_k_keyword, plan.retrieval_signals
                    )
                    fut_dense = executor.submit(
                        self._search_dense, plan.normalized_query, c_doc_ids, plan.top_k_dense
                    )
                    kw_hits = fut_kw.result()
                    dense_hits = fut_dense.result()
                    if kw_hits:
                        all_candidate_runs.append(kw_hits)
                        keyword_count += len(kw_hits)
                    if dense_hits:
                        all_candidate_runs.append(dense_hits)
                        dense_count += len(dense_hits)
            else:
                # Add dense runs additively to existing candidate runs
                dense_hits = self._search_dense(plan.normalized_query, c_doc_ids, plan.top_k_dense)
                if dense_hits:
                    all_candidate_runs.append(dense_hits)
                    dense_count += len(dense_hits)

            latencies["hybrid_retrieval_ms"] = (time.perf_counter() - t_ret) * 1000

            # Additive fusion with exact match protection
            fused = reciprocal_rank_fusion(
                all_candidate_runs,
                k=60,
                top_n=max(15, plan.top_k * 2),
                exact_identifiers=exact_ids,
            )
            fused_count = len(fused)

            # Measure confidence on fused candidates
            conf = self.confidence_scorer.score_candidates(
                query=plan.normalized_query,
                candidates=fused,
                execution_mode="balanced",
                exact_identifiers=exact_ids,
            )

            # Conditional reranking: bypass if exact match is strong, rerank if ambiguous
            t_rerank = time.perf_counter()
            should_rerank = (
                plan.requires_reranking
                and not (conf.top_score >= 8.0 and conf.term_coverage >= 0.7)
                and self.reranker.should_rerank(
                    query_type=plan.query_type,
                    execution_mode="balanced",
                    candidates=fused,
                )
            )

            if should_rerank:
                reranked = self.reranker.rerank(
                    query=plan.normalized_query, candidates=fused, top_k=plan.rerank_top_k
                )
                did_rerank = True
            else:
                reranked = self.reranker.bypass_rerank(fused, top_k=plan.top_k)

            latencies["reranking_ms"] = (time.perf_counter() - t_rerank) * 1000
            reranked_count = len(reranked)

            ctx = self.context_filter.filter_context(reranked, max_tokens=plan.context_budget)
            v_bal = self.evidence_gate.evaluate(plan.normalized_query, ctx)

            if v_bal.allowed:
                final_context = ctx
                verdict = v_bal
            else:
                if mode != "balanced" and (plan.complexity in ("medium", "high") or plan.sub_queries):
                    escalation_reason = f"Balanced evidence gate failed: {v_bal.reason}"
                    active_mode = "deep"
                else:
                    final_context = ctx
                    verdict = v_bal

        # -----------------------------------------------------------------
        # STAGE 3: DEEP ROUTE (Additive Expansion: Decomposition & Parent Chunks)
        # -----------------------------------------------------------------
        if active_mode == "deep" and (verdict is None or not verdict.allowed):
            cascade_stages.append("deep_expansion")
            t_ret = time.perf_counter()

            sub_queries = plan.sub_queries or [plan.normalized_query]
            if len(sub_queries) > 1:
                did_decompose = True

            # Ensure base keyword search is present in candidate pool
            if not any("bm25" in getattr(r[0], "retriever_name", "") for r in all_candidate_runs if r):
                base_kw = self._search_keyword(
                    plan.normalized_query, c_doc_ids, plan.top_k_keyword, plan.retrieval_signals
                )
                if base_kw:
                    all_candidate_runs.append(base_kw)
                    keyword_count += len(base_kw)

            # Sub-query searches added additively
            for sub_q in sub_queries[:3]:
                if plan.requires_parent_child:
                    pc_resp = self.parent_child_retriever.retrieve(
                        query=sub_q,
                        candidate_document_ids=c_doc_ids,
                        top_k=plan.top_k_dense,
                        expand_parent=True,
                    )
                    did_parent_expand = True
                    # Add child hits as anchor AND expanded parent hits
                    child_runs = [
                        SearchResult(
                            chunk_id=c.chunk_id,
                            document_id=c.document_id,
                            text=c.text,
                            score=c.score,
                            rank=c.rank,
                            parent_id=c.parent_id,
                            source_locator=c.source_locator,
                            page=c.page,
                            section=c.section,
                            chunk_type="child",
                            retriever_name="child_dense",
                            metadata=c.metadata,
                        )
                        for c in pc_resp.results
                    ]
                    parent_runs = [
                        SearchResult(
                            chunk_id=f"parent_{c.parent_id or c.chunk_id}",
                            document_id=c.document_id,
                            text=c.context_text,
                            score=c.score * 0.95,
                            rank=c.rank,
                            parent_id=c.parent_id,
                            source_locator=c.source_locator,
                            page=c.page,
                            section=c.section,
                            chunk_type="parent",
                            retriever_name="parent_dense",
                            metadata=c.metadata,
                        )
                        for c in pc_resp.results if c.expanded
                    ]
                    if child_runs:
                        all_candidate_runs.append(child_runs)
                    if parent_runs:
                        all_candidate_runs.append(parent_runs)
                else:
                    sub_dense = self._search_dense(sub_q, c_doc_ids, plan.top_k_dense)
                    if sub_dense:
                        all_candidate_runs.append(sub_dense)

            latencies["deep_retrieval_ms"] = (time.perf_counter() - t_ret) * 1000

            # Global additive fusion combining ALL accumulated runs
            deep_fused = reciprocal_rank_fusion(
                all_candidate_runs,
                k=60,
                top_n=plan.top_k * 2,
                exact_identifiers=exact_ids,
            )
            fused_count = len(deep_fused)

            t_rerank = time.perf_counter()
            reranked = self.reranker.rerank(
                query=plan.normalized_query,
                candidates=deep_fused,
                top_k=plan.rerank_top_k or 8,
            )
            did_rerank = True
            latencies["reranking_ms"] = (time.perf_counter() - t_rerank) * 1000
            reranked_count = len(reranked)

            final_context = self.context_filter.filter_context(reranked, max_tokens=plan.context_budget)
            verdict = self.evidence_gate.evaluate(plan.normalized_query, final_context)

        # Fallback verdict if context is empty
        if verdict is None:
            verdict = self.evidence_gate.evaluate(plan.normalized_query, final_context)

        is_escalated = len(cascade_stages) > 1

        retrieval_meta = RetrievalMetadata(
            strategy=plan.query_type,
            query_type=plan.query_type,
            execution_mode=active_mode,
            cascade_stages_executed=cascade_stages,
            escalated=is_escalated,
            escalation_reason=escalation_reason,
            reranked=did_rerank,
            parent_expanded=did_parent_expand,
            multi_query_used=did_multi_query,
            decomposed=did_decompose,
            dense_candidates=dense_count,
            keyword_candidates=keyword_count,
            multi_query_candidates=mq_count,
            fused_candidates=fused_count,
            reranked_candidates=reranked_count,
            filtered_context_candidates=len(final_context),
            latency_breakdown_ms=latencies,
        )

        # Early abstention if evidence is insufficient
        if not verdict.allowed:
            total_latency = (time.perf_counter() - start_time) * 1000
            latencies["total_ms"] = total_latency
            return RAGResponse(
                query=query,
                answer=(
                    f"I cannot provide a grounded answer based on the available sources: "
                    f"{verdict.reason}"
                ),
                confidence=verdict.confidence,
                abstained=True,
                abstention_reason=verdict.reason,
                query_type=plan.query_type,
                execution_mode=active_mode,
                sources=[],
                retrieval=retrieval_meta,
                latency_ms=total_latency,
            )

        # Answer generation with citation enforcement
        t_gen = time.perf_counter()
        gen_result = self.generator.generate(
            query=query, supporting_chunks=verdict.supporting_chunks
        )
        latencies["generation_ms"] = (time.perf_counter() - t_gen) * 1000

        total_latency = (time.perf_counter() - start_time) * 1000
        latencies["total_ms"] = total_latency

        return RAGResponse(
            query=query,
            answer=gen_result["answer"],
            confidence=verdict.confidence,
            abstained=False,
            query_type=plan.query_type,
            execution_mode=active_mode,
            sources=gen_result["sources"],
            retrieval=retrieval_meta,
            citation_validation=gen_result["validation"],
            latency_ms=total_latency,
        )
