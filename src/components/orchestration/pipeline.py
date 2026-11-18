from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

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
from ..retrieval.fusion import UnifiedCandidate, fuse_candidates
from ..retrieval.hybrid import HybridRetriever
from ..retrieval.iterative import BoundedIterativeRetriever
from ..retrieval.keyword import KeywordRetriever
from ..retrieval.multi_query import MultiQueryRetriever
from ..retrieval.parent_child import ParentChildRetriever
from .context_filter import ContextFilter
from .contracts import RAGResponse, RetrievalMetadata
from .evidence_gate import EvidenceGate, EvidenceVerdict
from .rerank_stage import RerankStage
from .retriever_executor import RetrieverExecutor

try:
    from config.models import AppConfig, get_config
except ImportError:
    from ....config.models import AppConfig, get_config

from ..registry import SystemRegistry

logger = logging.getLogger(__name__)

__all__ = ["RAGOrchestrator", "RAGResponse", "RetrievalMetadata"]


class RAGOrchestrator:
    """
    Production-Oriented Adaptive RAG Orchestrator.
    Coordinates decoupled query planning, multi-engine retrieval, conditional reranking,
    evidence gating, grounded synthesis, and citation repair.
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
        config: Optional[AppConfig] = None,
    ) -> None:
        self.config = config or get_config()
        self.doc_store = document_store or SystemRegistry.get_doc_store()
        self.planner = query_planner or QueryPlanner(document_store=self.doc_store)
        self.embedder = embedder or SystemRegistry.get_embedder()
        self.qdrant = qdrant_indexer or SystemRegistry.get_qdrant()
        self.bm25 = bm25_indexer or SystemRegistry.get_bm25()

        # Synchronize in-memory Qdrant with indexed BM25 chunks if Qdrant is empty
        if self.qdrant.count() == 0 and self.bm25.count() > 0:
            logger.info("Synchronizing in-memory Qdrant with indexed BM25 chunks...")
            import hashlib
            import re as _re
            from ..chunking.models import Chunk
            loaded = [Chunk.from_dict(c) for c in self.bm25._all_chunks]
            import numpy as np
            from pathlib import Path
            emb_rev = getattr(self.embedder, "cache_revision", f"{type(self.embedder).__name__}_{self.embedder.dimension}")
            safe_rev = _re.sub(r"[^a-zA-Z0-9_.-]+", "_", emb_rev)
            cache_file = Path(__file__).resolve().parents[3] / "metadata" / f"chunk_embeddings_{safe_rev}.npz"
            cached_dict = {}
            if cache_file.exists():
                try:
                    data = np.load(cache_file)
                    cached_dict = {k: data[k] for k in data.files}
                except Exception as c_exc:
                    logger.debug(f"Could not load vector cache {cache_file}: {c_exc}")

            def _chunk_cache_key(ch: Chunk) -> str:
                h = hashlib.sha256((ch.text or "").encode("utf-8")).hexdigest()[:16]
                return f"{ch.chunk_id}__{h}"

            missing = [c for c in loaded if _chunk_cache_key(c) not in cached_dict]
            if missing:
                new_vecs = self.embedder.embed_texts([c.text for c in missing])
                for c, v in zip(missing, new_vecs):
                    cached_dict[_chunk_cache_key(c)] = np.array(v, dtype=np.float32)
                try:
                    np.savez_compressed(cache_file, **cached_dict)
                except Exception as s_exc:
                    logger.debug(f"Could not save vector cache: {s_exc}")
            vecs = [cached_dict[_chunk_cache_key(c)].tolist() for c in loaded]
            self.qdrant.upsert_chunks(loaded, vecs)

        self.dense_retriever = DenseRetriever(indexer=self.qdrant, embedder=self.embedder)
        self.keyword_retriever = KeywordRetriever(indexer=self.bm25)
        self.hybrid_retriever = HybridRetriever(
            dense_retriever=self.dense_retriever,
            keyword_retriever=self.keyword_retriever,
            fusion_method=self.config.fusion_method,
        )
        self.parent_child_retriever = ParentChildRetriever(
            base_retriever=self.dense_retriever, document_store=self.doc_store
        )
        self.multi_query_retriever = MultiQueryRetriever(base_retriever=self.dense_retriever)
        self.iterative_retriever = BoundedIterativeRetriever(
            dense_retriever=self.dense_retriever,
            keyword_retriever=self.keyword_retriever,
            max_hops=2,
            top_k_per_hop=6,
        )

        self.reranker = reranker or SystemRegistry.get_reranker()
        self.context_filter = context_filter or ContextFilter()
        self.evidence_gate = evidence_gate or EvidenceGate()
        self.generator = generator or GroundedGenerator()
        self.confidence_scorer = confidence_scorer or RetrievalConfidenceScorer(
            confidence_high_threshold=self.config.confidence_high_threshold,
            confidence_medium_threshold=self.config.confidence_medium_threshold,
        )

        # Decoupled stage executors
        self.retriever_executor = RetrieverExecutor(
            qdrant_indexer=self.qdrant,
            bm25_indexer=self.bm25,
            dense_retriever=self.dense_retriever,
            keyword_retriever=self.keyword_retriever,
            cache_enabled=self.config.cache_enabled,
            fusion_method=self.config.fusion_method,
        )
        self.rerank_stage = RerankStage(reranker=self.reranker)

    def _search_keyword(
        self,
        query: str,
        candidate_document_ids: Optional[List[str]],
        top_k: int,
        signals: Dict[str, Any],
        metadata_filter: Optional[Any] = None,
    ) -> List[SearchResult]:
        return self.retriever_executor.search_keyword(
            query=query,
            candidate_document_ids=candidate_document_ids,
            top_k=top_k,
            signals=signals,
            metadata_filter=metadata_filter,
        )

    def _search_dense(
        self,
        query: str,
        candidate_document_ids: Optional[List[str]],
        top_k: int,
        metadata_filter: Optional[Any] = None,
    ) -> List[SearchResult]:
        return self.retriever_executor.search_dense(
            query=query,
            candidate_document_ids=candidate_document_ids,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

    def _consolidate_multi_hop_candidates(
        self,
        reranked_candidates: List[RerankerResult],
        hop_anchor_ids: Set[str],
        num_hops: int = 1,
        min_score_ratio: float = 0.55,
        target_top_k: int = 3,
    ) -> List[RerankerResult]:
        """Post-Hop Global Consolidation & Pruning with Hop Coverage Preservation."""
        if not reranked_candidates:
            return []
        top_score = max((r.score for r in reranked_candidates), default=1.0)
        score_threshold = top_score * min_score_ratio

        anchors = [r for r in reranked_candidates if r.chunk_id in hop_anchor_ids]
        non_anchors = [
            r for r in reranked_candidates
            if r.chunk_id not in hop_anchor_ids and r.score >= score_threshold
        ]
        max_final_chunks = max(target_top_k, num_hops + 1)

        seen_chunk_ids = {r.chunk_id for r in anchors}
        combined = list(anchors)
        for r in non_anchors:
            if len(combined) >= max_final_chunks:
                break
            if r.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(r.chunk_id)
                combined.append(r)

        combined.sort(key=lambda r: r.score, reverse=True)
        for rank, r in enumerate(combined, start=1):
            r.rank = rank
        return combined

    def _retrieve_and_gate(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        mode: Optional[str] = None,
        candidate_document_ids: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        enable_consensus_guard: Optional[bool] = None,
        intent_aware_rerank_budget: bool = False,
    ) -> Tuple[QueryPlan, RetrievalMetadata, EvidenceVerdict, Dict[str, float], float]:
        start_time = time.perf_counter()
        latencies: Dict[str, float] = {}

        effective_doc_ids = document_ids or candidate_document_ids

        # Stage 1: Tiered Query Planning & Scope Resolution
        t_plan = time.perf_counter()
        plan = self.planner.plan(
            query,
            explicit_document_ids=effective_doc_ids,
            forced_mode=mode if mode in ("fast", "balanced", "deep") else None,
        )
        latencies["planning_ms"] = (time.perf_counter() - t_plan) * 1000

        # Scope Invariant: Explicit caller filters are strictly isolated. Never relaxed.
        if effective_doc_ids:
            c_doc_ids = effective_doc_ids
            is_explicit = True
        elif plan.is_explicit_filter and plan.explicit_document_ids:
            c_doc_ids = plan.explicit_document_ids
            is_explicit = True
        else:
            c_doc_ids = None
            is_explicit = False

        m_filter = metadata_filter or (
            plan.retrieval_filter if (is_explicit and getattr(self.config, "metadata_filter_enabled", True)) else None
        )
        active_mode = mode or plan.execution_mode
        exact_ids = plan.retrieval_signals.get("exact_identifiers", [])

        # Stage 2: Independent Retrieval Execution
        ret_exec_mode = "fast" if active_mode in ("fast", "bm25") else ("dense" if active_mode == "dense" else "hybrid")
        ret_result = self.retriever_executor.execute_retrieval(
            query=plan.normalized_query,
            execution_mode=ret_exec_mode,
            top_k_keyword=plan.top_k_keyword,
            top_k_dense=plan.top_k_dense,
            retrieval_signals=plan.retrieval_signals,
            candidate_document_ids=c_doc_ids,
            metadata_filter=m_filter,
            inferred_document_ids=plan.inferred_document_ids,
            exact_identifiers=exact_ids,
        )
        latencies.update(ret_result.stage_latencies)

        # Stage 3: Retrieval Quality & Evidence Confidence Assessment
        t_conf = time.perf_counter()
        conf = self.confidence_scorer.score_candidates(
            query=plan.normalized_query,
            candidates=ret_result.candidates,
            execution_mode=active_mode,
            exact_identifiers=exact_ids,
        )
        latencies["confidence_ms"] = (time.perf_counter() - t_conf) * 1000

        # Stage 4: Conditional Reranking (Consensus Guard + Micro-Rerank)
        top_bm25_id = ret_result.raw_bm25_hits[0].chunk_id if ret_result.raw_bm25_hits else None
        top_dense_id = ret_result.raw_dense_hits[0].chunk_id if ret_result.raw_dense_hits else None

        rerank_res = self.rerank_stage.execute_rerank(
            query=plan.normalized_query,
            candidates=ret_result.candidates,
            mode=active_mode,
            confidence_score=conf.score,
            top_bm25_id=top_bm25_id,
            top_dense_id=top_dense_id,
            exact_identifiers=exact_ids,
            rerank_top_k=plan.rerank_top_k,
            default_top_k=plan.top_k,
            enable_consensus_guard=enable_consensus_guard,
            query_type=plan.query_type,
            intent_aware_budget=intent_aware_rerank_budget,
        )
        latencies["reranking_ms"] = rerank_res.rerank_latency_ms
        active_candidates = rerank_res.candidates

        # Stage 5: Context Assembly & Parent-Child Expansion
        did_parent_expand = False
        if plan.requires_parent_child:
            t_pe = time.perf_counter()
            active_candidates = self.parent_child_retriever.expand_candidates(
                active_candidates,
                max_parents=plan.max_parents,
                max_parent_tokens=plan.max_parent_tokens,
            )
            did_parent_expand = True
            latencies["parent_expansion_ms"] = (time.perf_counter() - t_pe) * 1000

        t_cf = time.perf_counter()
        final_context = self.context_filter.filter_context(active_candidates, max_tokens=plan.context_budget)
        latencies["context_filter_ms"] = (time.perf_counter() - t_cf) * 1000

        # Stage 6: Evidence Gate & Groundedness Verification
        t_eg = time.perf_counter()
        verdict = self.evidence_gate.evaluate(plan.normalized_query, final_context)
        latencies["evidence_gate_ms"] = (time.perf_counter() - t_eg) * 1000

        # Cross-Encoder Recovery: If reranking was bypassed but evidence gate failed, recover with reranking
        if not verdict.allowed and not rerank_res.did_rerank and active_mode not in ("fast", "bm25"):
            t_rec = time.perf_counter()
            recovered = self.reranker.rerank(
                query=plan.normalized_query,
                candidates=ret_result.candidates,
                top_k=plan.rerank_top_k,
            )
            latencies["reranker_recovery_ms"] = (time.perf_counter() - t_rec) * 1000
            if plan.requires_parent_child:
                recovered = self.parent_child_retriever.expand_candidates(
                    recovered, max_parents=plan.max_parents, max_parent_tokens=plan.max_parent_tokens
                )
            final_context = self.context_filter.filter_context(recovered, max_tokens=plan.context_budget)
            verdict = self.evidence_gate.evaluate(plan.normalized_query, final_context)

        parent_exp_count = sum(
            1 for c in final_context
            if getattr(c, "expanded", False)
            or (isinstance(getattr(c, "metadata", None), dict) and c.metadata.get("expanded_parent"))
        )

        cascade_stages = [ret_result.strategy]
        if rerank_res.did_rerank:
            cascade_stages.append("cross_encoder_rerank")
        if not verdict.allowed:
            cascade_stages.append("abstention_gated")

        # Record pre-generation retrieved candidates so retrieval evaluation is
        # strictly decoupled from downstream evidence gating and citation validation.
        pre_gen_candidates = [
            {
                "rank": idx,
                "chunk_id": getattr(c, "chunk_id", ""),
                "document_id": getattr(c, "document_id", ""),
                "text": getattr(c, "text", ""),
                "score": float(getattr(c, "rerank_score", None) or getattr(c, "score", 0.0) or 0.0),
                "source_locator": getattr(c, "source_locator", None),
            }
            for idx, c in enumerate(final_context, start=1)
        ]

        retrieval_meta = RetrievalMetadata(
            strategy=plan.query_type,
            query_type=plan.query_type,
            execution_mode=active_mode,
            cascade_stages_executed=cascade_stages,
            reranked=rerank_res.did_rerank,
            parent_expanded=did_parent_expand,
            dense_candidates=len(ret_result.raw_dense_hits),
            keyword_candidates=len(ret_result.raw_bm25_hits),
            fused_candidates=len(ret_result.candidates),
            reranked_candidates=len(active_candidates),
            rerank_pool_size=rerank_res.rerank_pool_size,
            parent_expanded_count=parent_exp_count,
            filtered_context_candidates=len(final_context),
            confidence_score=verdict.confidence,
            confidence_tier=conf.tier if conf else None,
            confidence_signals=verdict.signals or (conf.signals if conf else {}),
            latency_breakdown_ms=latencies,
            retrieved_candidates=pre_gen_candidates,
        )

        return plan, retrieval_meta, verdict, latencies, start_time

    def _abstention_response(
        self,
        query: str,
        plan: QueryPlan,
        retrieval_meta: RetrievalMetadata,
        verdict: EvidenceVerdict,
        latencies: Dict[str, float],
        start_time: float,
        citation_validation: Optional[Dict[str, Any]] = None,
    ) -> RAGResponse:
        total_latency = (time.perf_counter() - start_time) * 1000
        latencies["generation_ms"] = latencies.get("generation_ms", 0.0)
        latencies["retrieval_ms"] = max(0.0, total_latency - latencies.get("planning_ms", 0.0) - latencies["generation_ms"])
        latencies["total_ms"] = total_latency
        return RAGResponse(
            query=query,
            answer=f"I cannot provide a grounded answer based on the available sources: {verdict.reason}",
            confidence=verdict.confidence,
            abstained=True,
            abstention_reason=verdict.reason,
            query_type=plan.query_type,
            execution_mode=retrieval_meta.execution_mode,
            sources=[],
            retrieval=retrieval_meta,
            citation_validation=citation_validation or {},
            latency_ms=total_latency,
        )

    def query(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        mode: Optional[str] = None,
        candidate_document_ids: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        enable_consensus_guard: Optional[bool] = None,
        intent_aware_rerank_budget: bool = False,
    ) -> RAGResponse:
        """Execute a full production RAG query with strict evidence gating and citation repair."""
        plan, retrieval_meta, verdict, latencies, start_time = self._retrieve_and_gate(
            query=query,
            document_ids=document_ids,
            mode=mode,
            candidate_document_ids=candidate_document_ids,
            metadata_filter=metadata_filter,
            enable_consensus_guard=enable_consensus_guard,
            intent_aware_rerank_budget=intent_aware_rerank_budget,
        )

        if not verdict.allowed:
            return self._abstention_response(query, plan, retrieval_meta, verdict, latencies, start_time)

        t_gen = time.perf_counter()
        gen_result = self.generator.generate(
            query=query, supporting_chunks=verdict.supporting_chunks
        )
        latencies["generation_ms"] = (time.perf_counter() - t_gen) * 1000

        # Objective 5: If answer fails citation validation/grounding after repair, trigger abstention
        val_meta = gen_result.get("validation", {})
        if val_meta.get("needs_abstention"):
            verdict_rej = EvidenceVerdict(
                allowed=False,
                confidence=verdict.confidence,
                reason=val_meta.get("details") or "Answer failed factual citation verification.",
            )
            return self._abstention_response(
                query,
                plan,
                retrieval_meta,
                verdict_rej,
                latencies,
                start_time,
                citation_validation=val_meta,
            )

        total_latency = (time.perf_counter() - start_time) * 1000
        latencies["retrieval_ms"] = max(
            0.0, total_latency - latencies.get("planning_ms", 0.0) - latencies["generation_ms"]
        )
        latencies["total_ms"] = total_latency

        return RAGResponse(
            query=query,
            answer=gen_result["answer"],
            confidence=verdict.confidence,
            abstained=False,
            query_type=plan.query_type,
            execution_mode=retrieval_meta.execution_mode,
            sources=gen_result["sources"],
            retrieval=retrieval_meta,
            citation_validation=gen_result["validation"],
            latency_ms=total_latency,
        )

    def query_stream(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        mode: Optional[str] = None,
        candidate_document_ids: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield SSE-ready stage, delta, and done events for /query/stream."""
        plan, retrieval_meta, verdict, latencies, start_time = self._retrieve_and_gate(
            query=query,
            document_ids=document_ids,
            mode=mode,
            candidate_document_ids=candidate_document_ids,
            metadata_filter=metadata_filter,
        )
        retrieval_ms = (time.perf_counter() - start_time) * 1000

        yield {
            "event": "retrieval",
            "confidence": verdict.confidence,
            "execution_mode": retrieval_meta.execution_mode,
            "query_type": plan.query_type,
            "latency_retrieval_ms": round(retrieval_ms, 2),
        }

        if not verdict.allowed:
            resp = self._abstention_response(query, plan, retrieval_meta, verdict, latencies, start_time)
            yield {"event": "delta", "text": resp.answer}
            yield {"event": "done", **resp.model_dump()}
            return

        t_gen = time.perf_counter()
        gen_result: Dict[str, Any] = {"answer": "", "sources": [], "validation": {}}
        for kind, payload in self.generator.generate_stream(
            query=query, supporting_chunks=verdict.supporting_chunks
        ):
            if kind == "delta":
                yield {"event": "delta", "text": payload}
            elif kind == "final":
                gen_result = payload

        latencies["generation_ms"] = (time.perf_counter() - t_gen) * 1000
        total_latency = (time.perf_counter() - start_time) * 1000
        latencies["total_ms"] = total_latency

        val_meta = gen_result.get("validation", {})
        if val_meta.get("needs_abstention"):
            resp = self._abstention_response(
                query,
                plan,
                retrieval_meta,
                EvidenceVerdict(allowed=False, confidence=verdict.confidence, reason=val_meta.get("details", "")),
                latencies,
                start_time,
            )
            yield {"event": "done", **resp.model_dump()}
            return

        resp = RAGResponse(
            query=query,
            answer=gen_result["answer"],
            confidence=verdict.confidence,
            abstained=False,
            query_type=plan.query_type,
            execution_mode=retrieval_meta.execution_mode,
            sources=gen_result["sources"],
            retrieval=retrieval_meta,
            citation_validation=gen_result["validation"],
            latency_ms=total_latency,
        )
        yield {"event": "done", **resp.model_dump()}


# Aliases for architectural clarity
AdaptiveRAGPipeline = RAGOrchestrator
RAGPipeline = RAGOrchestrator

