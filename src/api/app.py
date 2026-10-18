from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..components.chunking import ChunkingPipeline
from ..components.indexing import (
    BM25Indexer,
    DocumentStore,
    IndexingPipeline,
    QdrantIndexer,
    SentenceTransformerEmbedder,
)
from ..components.ingestion.config import IngestionConfig
from ..components.ingestion.pipeline import IngestionPipeline as DocIngestionPipeline
from ..components.orchestration import RAGOrchestrator
from ..components.registry import SystemRegistry
from .models import (
    DocumentItem,
    HealthResponse,
    IngestJobResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SearchItem,
    SearchRequest,
    SearchResponse,
    SourceReference,
)

logger = logging.getLogger(__name__)

DOCUMENTS_DIR = Path(__file__).resolve().parents[2] / "documents"
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Production-oriented Adaptive RAG API",
    description="Production-Grade, Modular, Evidence-Grounded RAG Platform",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")


class SystemState:
    """Manages shared application dependencies using SystemRegistry."""

    def __init__(self) -> None:
        self.doc_store = SystemRegistry.get_doc_store()
        self.embedder = SystemRegistry.get_embedder()
        self.qdrant = SystemRegistry.get_qdrant()
        self.bm25 = SystemRegistry.get_bm25()
        self.chunker = ChunkingPipeline()
        self.indexing_pipe = IndexingPipeline(
            embedder=self.embedder,
            qdrant_indexer=self.qdrant,
            bm25_indexer=self.bm25,
            document_store=self.doc_store,
            chunking_pipeline=self.chunker,
        )
        self.ingestion_pipe = DocIngestionPipeline(IngestionConfig())
        self.orchestrator = RAGOrchestrator(
            embedder=self.embedder,
            qdrant_indexer=self.qdrant,
            bm25_indexer=self.bm25,
            document_store=self.doc_store,
        )
        self.jobs: Dict[str, Dict[str, Any]] = {}


state = SystemState()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    qdr_healthy = state.qdrant.health_check()
    docs = state.doc_store.list_documents()
    return HealthResponse(
        status="healthy" if qdr_healthy else "degraded",
        qdrant_healthy=qdr_healthy,
        document_count=len(docs),
        chunk_count=state.qdrant.count(),
    )


@app.get("/ready")
def ready() -> Dict[str, str]:
    if state.qdrant.health_check():
        return {"status": "ready"}
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Qdrant not ready")


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    docs = state.doc_store.list_documents()
    return {
        "documents_total": len(docs),
        "indexed_documents": sum(1 for d in docs if d.status == "INDEXED"),
        "qdrant_points_count": state.qdrant.count(),
        "bm25_chunks_count": state.bm25.count(),
        "active_jobs_count": len(state.jobs),
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty."
        )

    try:
        res = state.orchestrator.query(
            query=request.query,
            document_ids=request.document_ids,
            mode=request.mode,
        )

        sources = [
            SourceReference(
                citation_id=s.get("citation_id", 0),
                document_id=s.get("document_id", ""),
                filename=s.get("filename", ""),
                chunk_id=s.get("chunk_id", ""),
                page=s.get("page"),
                section=s.get("section"),
                source_locator=s.get("source_locator"),
                snippet=s.get("snippet", ""),
            )
            for s in res.sources
        ]

        return QueryResponse(
            query=res.query,
            answer=res.answer,
            confidence=res.confidence,
            abstained=res.abstained,
            abstention_reason=res.abstention_reason,
            query_type=res.query_type,
            execution_mode=res.execution_mode,
            sources=sources,
            retrieval=res.retrieval.model_dump(),
            citation_validation=res.citation_validation,
            latency_ms=res.latency_ms,
        )
    except Exception as exc:
        logger.error(f"Error handling query: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the query: {str(exc)}",
        )


@app.post("/query/stream")
async def query_stream(request: QueryRequest) -> StreamingResponse:
    """
    Streaming SSE endpoint for grounded RAG queries.
    Emits three event types:
      - ``retrieval``: Sent after retrieval completes with confidence and source metadata.
      - ``delta``: One event per 6-char chunk of the generated answer for progressive rendering.
      - ``done``: Final event with the full structured payload, citations, and latency breakdown.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty."
        )

    async def _event_stream() -> AsyncGenerator[str, None]:
        import asyncio
        import concurrent.futures

        loop = asyncio.get_event_loop()

        def _run_query():
            return state.orchestrator.query(
                query=request.query,
                document_ids=request.document_ids,
                mode=request.mode,
            )

        try:
            # Run the blocking orchestrator call in a thread pool to keep the event loop free
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                res = await loop.run_in_executor(pool, _run_query)

            # Event 1: Retrieval metadata — emitted before token streaming starts
            retrieval_event = {
                "event": "retrieval",
                "confidence": res.confidence,
                "execution_mode": res.execution_mode,
                "query_type": res.query_type,
                "abstained": res.abstained,
                "abstention_reason": res.abstention_reason,
                "source_count": len(res.sources),
                "latency_retrieval_ms": res.retrieval.latency_breakdown_ms.get(
                    "hybrid_retrieval_ms",
                    res.retrieval.latency_breakdown_ms.get("fast_retrieval_ms", 0.0),
                ),
            }
            yield f"data: {json.dumps(retrieval_event)}\n\n"

            # Event 2: Delta tokens — stream the answer text in 6-character word-group chunks
            answer = res.answer or ""
            chunk_size = 6
            for i in range(0, len(answer), chunk_size):
                yield f"data: {json.dumps({'event': 'delta', 'text': answer[i: i + chunk_size]})}\n\n"
                await asyncio.sleep(0)  # Yield to flush without blocking

            # Event 3: Done — full structured payload
            sources_out = [
                {
                    "citation_id": s.get("citation_id", 0),
                    "document_id": s.get("document_id", ""),
                    "filename": s.get("filename", ""),
                    "chunk_id": s.get("chunk_id", ""),
                    "page": s.get("page"),
                    "section": s.get("section"),
                    "snippet": s.get("snippet", ""),
                }
                for s in res.sources
            ]
            done_event = {
                "event": "done",
                "query": res.query,
                "answer": answer,
                "confidence": res.confidence,
                "abstained": res.abstained,
                "abstention_reason": res.abstention_reason,
                "sources": sources_out,
                "citation_validation": res.citation_validation,
                "latency_ms": res.latency_ms,
                "retrieval": res.retrieval.model_dump(),
            }
            yield f"data: {json.dumps(done_event)}\n\n"

        except Exception as exc:
            logger.error(f"Streaming query error: {exc}", exc_info=True)
            yield f"data: {json.dumps({'event': 'error', 'detail': str(exc)})}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty."
        )

    start_time = time.perf_counter()
    k = max(1, min(request.top_k, 50))
    results: List[SearchItem] = []

    if request.strategy == "keyword":
        hits = state.bm25.search(
            query=request.query,
            top_k=k,
            candidate_document_ids=request.document_ids,
            metadata_filter=request.metadata_filter,
        )
        for h in hits:
            results.append(
                SearchItem(
                    chunk_id=h.chunk_id,
                    document_id=h.document_id,
                    text=h.text,
                    score=h.score,
                    rank=h.rank,
                    source_locator=h.source_locator,
                    page=h.page,
                    section=h.section,
                    retriever="keyword_bm25",
                    metadata=h.metadata,
                )
            )
    elif request.strategy == "dense":
        dense_resp = state.orchestrator.dense_retriever.retrieve(
            query=request.query,
            candidate_document_ids=request.document_ids,
            metadata_filter=request.metadata_filter,
            top_k=k,
        )
        for h in dense_resp.results:
            results.append(
                SearchItem(
                    chunk_id=h.chunk_id,
                    document_id=h.document_id,
                    text=h.text,
                    score=h.score,
                    rank=h.rank,
                    source_locator=h.source_locator,
                    page=h.page,
                    section=h.section,
                    retriever="dense_qdrant",
                    metadata=h.metadata,
                )
            )
    else:  # hybrid
        resp = state.orchestrator.hybrid_retriever.retrieve(
            query=request.query,
            candidate_document_ids=request.document_ids,
            metadata_filter=request.metadata_filter,
            top_k=k,
        )
        for cand in resp.results:
            results.append(
                SearchItem(
                    chunk_id=cand.chunk_id,
                    document_id=cand.document_id,
                    text=cand.text,
                    score=cand.fusion_score,
                    rank=cand.rank,
                    source_locator=cand.source_locator,
                    page=cand.page,
                    section=cand.section,
                    retriever="parallel_hybrid_rrf",
                    metadata=cand.metadata,
                )
            )

    latency = (time.perf_counter() - start_time) * 1000
    return SearchResponse(
        query=request.query,
        total_results=len(results),
        results=results,
        latency_ms=latency,
    )


@app.get("/documents", response_model=List[DocumentItem])
def list_documents() -> List[DocumentItem]:
    docs = state.doc_store.list_documents()
    return [
        DocumentItem(
            document_id=d.document_id,
            filename=d.filename,
            sha256=d.sha256,
            status=d.status,
            chunk_count=d.chunk_count,
            size_bytes=d.size_bytes,
            indexed_at=d.indexed_at,
            metadata=d.metadata,
        )
        for d in docs
    ]


@app.get("/documents/{document_id}", response_model=DocumentItem)
def get_document(document_id: str) -> DocumentItem:
    doc = state.doc_store.get(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found.",
        )
    return DocumentItem(
        document_id=doc.document_id,
        filename=doc.filename,
        sha256=doc.sha256,
        status=doc.status,
        chunk_count=doc.chunk_count,
        size_bytes=doc.size_bytes,
        indexed_at=doc.indexed_at,
        metadata=doc.metadata,
    )


@app.delete("/documents/{document_id}")
def delete_document(document_id: str) -> Dict[str, Any]:
    doc = state.doc_store.get(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found.",
        )
    deleted = state.indexing_pipe.delete_document(document_id)
    return {"document_id": document_id, "deleted": deleted}


@app.post("/documents/upload", response_model=DocumentItem)
async def upload_document(file: UploadFile = File(...)) -> DocumentItem:
    filename = Path(file.filename).name
    if not filename or ".." in filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename."
        )

    target_path = DOCUMENTS_DIR / filename
    try:
        with open(target_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {exc}",
        )

    try:
        doc = state.ingestion_pipe.ingest_file(target_path)
        stats = state.indexing_pipe.index_documents([doc], force_reindex=True)
        rec = state.doc_store.get(doc.document_id)
        if not rec:
            raise RuntimeError("Document was not stored.")
        return DocumentItem(
            document_id=rec.document_id,
            filename=rec.filename,
            sha256=rec.sha256,
            status=rec.status,
            chunk_count=rec.chunk_count,
            size_bytes=rec.size_bytes,
            indexed_at=rec.indexed_at,
            metadata=rec.metadata,
        )
    except Exception as exc:
        logger.error(f"Error ingesting uploaded document: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {exc}",
        )


def _run_background_ingest(job_id: str, directory: Path, force_reindex: bool):
    try:
        state.jobs[job_id]["status"] = "PROCESSING"
        files = [p for p in directory.rglob("*") if p.is_file()]
        ingest_result = state.ingestion_pipe.ingest_many(files, strict=False)
        index_stats = state.indexing_pipe.index_documents(
            ingest_result.documents, force_reindex=force_reindex
        )
        state.jobs[job_id]["status"] = "COMPLETED"
        state.jobs[job_id]["result"] = {
            "documents_indexed": index_stats.documents_indexed,
            "documents_skipped": index_stats.documents_skipped,
            "chunks_indexed": index_stats.chunks_indexed,
            "errors": ingest_result.errors + index_stats.errors,
        }
    except Exception as exc:
        logger.error(f"Background ingest job {job_id} failed: {exc}")
        state.jobs[job_id]["status"] = "FAILED"
        state.jobs[job_id]["error"] = str(exc)


@app.post(
    "/documents/ingest/async",
    response_model=IngestJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def ingest_directory_async(
    request: IngestRequest, background_tasks: BackgroundTasks
) -> IngestJobResponse:
    target_dir = Path(request.directory) if request.directory else DOCUMENTS_DIR
    if not target_dir.exists() or not target_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Directory not found: {target_dir}",
        )

    job_id = str(uuid.uuid4())[:8]
    state.jobs[job_id] = {
        "job_id": job_id,
        "status": "QUEUED",
        "created_at": time.time(),
    }
    background_tasks.add_task(
        _run_background_ingest, job_id, target_dir, request.force_reindex
    )
    return IngestJobResponse(
        job_id=job_id,
        status="QUEUED",
        message="Background ingestion job submitted successfully.",
    )


@app.get("/documents/jobs/{job_id}")
def get_job_status(job_id: str) -> Dict[str, Any]:
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )
    return job


@app.post("/documents/ingest", response_model=IngestResponse)
def ingest_directory(request: IngestRequest) -> IngestResponse:
    target_dir = Path(request.directory) if request.directory else DOCUMENTS_DIR
    if not target_dir.exists() or not target_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Directory not found: {target_dir}",
        )

    files = [p for p in target_dir.rglob("*") if p.is_file()]
    if not files:
        return IngestResponse(
            documents_indexed=0,
            documents_skipped=0,
            chunks_indexed=0,
            duration_ms=0.0,
            errors=[],
        )

    start = time.perf_counter()
    ingest_result = state.ingestion_pipe.ingest_many(files, strict=False)
    index_stats = state.indexing_pipe.index_documents(
        ingest_result.documents, force_reindex=request.force_reindex
    )

    duration = (time.perf_counter() - start) * 1000
    errors = ingest_result.errors + index_stats.errors

    return IngestResponse(
        documents_indexed=index_stats.documents_indexed,
        documents_skipped=index_stats.documents_skipped + len(ingest_result.skipped),
        chunks_indexed=index_stats.chunks_indexed,
        duration_ms=duration,
        errors=errors,
    )
