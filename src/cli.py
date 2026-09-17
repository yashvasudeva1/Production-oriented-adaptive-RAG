from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from .components.chunking import ChunkingPipeline
from .components.indexing import (
    BM25Indexer,
    DocumentStore,
    IndexingPipeline,
    QdrantIndexer,
    SentenceTransformerEmbedder,
)
from .components.ingestion.config import IngestionConfig
from .components.ingestion.pipeline import IngestionPipeline as DocIngestionPipeline
from .components.orchestration import RAGOrchestrator


def get_system():
    embedder = SentenceTransformerEmbedder()
    qdrant = QdrantIndexer(vector_size=embedder.dimension)
    bm25 = BM25Indexer()
    doc_store = DocumentStore()
    chunker = ChunkingPipeline()
    indexing_pipe = IndexingPipeline(
        embedder=embedder,
        qdrant_indexer=qdrant,
        bm25_indexer=bm25,
        document_store=doc_store,
        chunking_pipeline=chunker,
    )
    ingestion_pipe = DocIngestionPipeline(IngestionConfig())
    orchestrator = RAGOrchestrator(
        embedder=embedder,
        qdrant_indexer=qdrant,
        bm25_indexer=bm25,
        document_store=doc_store,
    )
    return ingestion_pipe, indexing_pipe, orchestrator, qdrant, bm25, doc_store


def cmd_health(args):
    _, _, _, qdrant, bm25, doc_store = get_system()
    q_ok = qdrant.health_check()
    docs = doc_store.list_documents()
    status = {
        "status": "healthy" if q_ok else "degraded",
        "qdrant_connected": q_ok,
        "indexed_documents": len(docs),
        "qdrant_points": qdrant.count(),
        "bm25_chunks": bm25.count(),
    }
    print(json.dumps(status, indent=2))


def cmd_ingest(args):
    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path does not exist: {path}", file=sys.stderr)
        sys.exit(1)

    ingestion_pipe, indexing_pipe, _, _, _, _ = get_system()

    if path.is_file():
        files = [path]
    else:
        files = [p for p in path.rglob("*") if p.is_file()]

    print(f"Ingesting {len(files)} files from {path}...")
    ingest_result = ingestion_pipe.ingest_many(files, strict=False)
    print(f"Parsed {len(ingest_result.documents)} documents. Indexing...")

    stats = indexing_pipe.index_documents(
        ingest_result.documents, force_reindex=args.force
    )
    print(
        f"Indexed {stats.documents_indexed} documents ({stats.chunks_indexed} chunks) "
        f"in {stats.duration_ms:.1f}ms. Skipped: {stats.documents_skipped}."
    )
    if stats.errors:
        print(f"Errors encountered: {len(stats.errors)}")
        for err in stats.errors:
            print(f" - {err}")


def cmd_search(args):
    _, _, orchestrator, _, bm25, _ = get_system()
    print(f"Searching for: '{args.query}' (strategy: {args.strategy})...")
    if args.strategy == "keyword":
        hits = bm25.search(args.query, top_k=args.top_k)
        for h in hits:
            print(f"[{h.rank}] score={h.score:.3f} | {h.source_locator} | {h.text[:120]}...")
    elif args.strategy == "dense":
        resp = orchestrator.dense_retriever.retrieve(args.query, top_k=args.top_k)
        for h in resp.results:
            print(f"[{h.rank}] score={h.score:.3f} | {h.source_locator} | {h.text[:120]}...")
    else:
        # Hybrid
        hits = bm25.search(args.query, top_k=args.top_k)
        for h in hits:
            print(f"[{h.rank}] score={h.score:.3f} | {h.source_locator} | {h.text[:120]}...")


def cmd_query(args):
    _, _, orchestrator, _, _, _ = get_system()
    mode_str = f" [mode: {args.mode}]" if getattr(args, "mode", None) else ""
    print(f"Executing RAG query: '{args.query}'{mode_str}...")
    res = orchestrator.query(args.query, mode=getattr(args, "mode", None))
    print("-" * 60)
    print(f"QUERY TYPE : {res.query_type}")
    print(f"EXEC MODE  : {res.execution_mode}")
    print(f"CONFIDENCE : {res.confidence:.2f}")
    print(f"ABSTAINED  : {res.abstained}")
    if res.abstained:
        print(f"REASON     : {res.abstention_reason}")
    print(f"LATENCY    : {res.latency_ms:.1f}ms")
    print("-" * 60)
    print("ANSWER:")
    print(res.answer)
    print("-" * 60)
    if res.sources:
        print(f"SOURCES ({len(res.sources)}):")
        for s in res.sources:
            print(f" [{s.get('citation_id')}] {s.get('filename')} (locator: {s.get('source_locator')})")
            print(f"     \"{s.get('snippet', '')}...\"")


def main():
    parser = argparse.ArgumentParser(
        description="ResearchLens — Production-Grade RAG CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # health
    subparsers.add_parser("health", help="Check system and index health")

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest and index documents")
    p_ingest.add_argument("path", help="Path to document file or directory")
    p_ingest.add_argument(
        "--force", action="store_true", help="Force reindexing existing documents"
    )

    # search
    p_search = subparsers.add_parser("search", help="Execute vector/keyword search")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument(
        "--strategy", choices=["hybrid", "dense", "keyword"], default="hybrid"
    )
    p_search.add_argument("--top-k", type=int, default=5)

    # query
    p_query = subparsers.add_parser("query", help="Execute grounded RAG query")
    p_query.add_argument("query", help="Natural language question")
    p_query.add_argument(
        "--mode", choices=["fast", "balanced", "deep"], default=None, help="Execution mode (fast, balanced, deep)"
    )

    args = parser.parse_args()
    if args.command == "health":
        cmd_health(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "query":
        cmd_query(args)


if __name__ == "__main__":
    main()
