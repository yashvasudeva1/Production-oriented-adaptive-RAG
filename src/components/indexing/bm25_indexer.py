from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..chunking.models import Chunk
from ..keyword_matching import BM25Config, BM25Retriever
from .models import SearchResult

logger = logging.getLogger(__name__)

DEFAULT_CHUNKS_PATH = (
    Path(__file__).resolve().parents[3] / "metadata" / "chunks.json"
)


class BM25Indexer:
    """
    Keyword indexer wrapping the existing BM25Retriever component.
    Provides persistence to metadata/chunks.json, chunk upserts,
    document deletion, and unified SearchResult outputs.
    """

    def __init__(
        self,
        chunks_path: Path | str = DEFAULT_CHUNKS_PATH,
        config: Optional[BM25Config] = None,
        index_path: Optional[Path | str] = None,
    ) -> None:
        actual_path = index_path or chunks_path
        self.chunks_path = Path(actual_path)
        self.chunks_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.chunks_path.exists():
            self.chunks_path.write_text("[]", encoding="utf-8")

        self.config = config or BM25Config(chunks_path=self.chunks_path)
        self._all_chunks: List[Dict[str, Any]] = self._load_disk_chunks()

        self.retriever = BM25Retriever(config=self.config)
        if self._all_chunks:
            self.retriever.build_from_chunks(self._all_chunks)

    def _load_disk_chunks(self) -> List[Dict[str, Any]]:
        if not self.chunks_path.exists():
            return []
        try:
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and "chunks" in data:
                    return data["chunks"]
        except Exception as exc:
            logger.error(f"Failed to load existing chunks from {self.chunks_path}: {exc}")
        return []

    def _save_disk_chunks(self) -> None:
        try:
            with open(self.chunks_path, "w", encoding="utf-8") as f:
                json.dump(self._all_chunks, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.error(f"Failed to persist chunks to {self.chunks_path}: {exc}")

    def upsert_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Upsert chunks into BM25 index and save to disk."""
        if not chunks:
            return 0

        chunk_dict_map = {c["chunk_id"]: c for c in self._all_chunks}
        for chunk in chunks:
            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "parent_id": chunk.parent_id,
                "text": chunk.text,
                "page": chunk.page,
                "section": chunk.section,
                "slide": chunk.slide,
                "sheet": chunk.sheet,
                "source_locator": chunk.source_locator,
                "chunk_type": chunk.chunk_type,
                "token_count": chunk.token_count,
                "metadata": chunk.metadata,
            }
            chunk_dict_map[chunk.chunk_id] = payload

        self._all_chunks = list(chunk_dict_map.values())
        self._save_disk_chunks()
        self.retriever.build_from_chunks(self._all_chunks)
        return len(chunks)

    def delete_by_document_id(self, document_id: str) -> bool:
        """Remove all chunks associated with document_id."""
        initial_len = len(self._all_chunks)
        self._all_chunks = [
            c for c in self._all_chunks if str(c.get("document_id")) != str(document_id)
        ]
        if len(self._all_chunks) != initial_len:
            self._save_disk_chunks()
            self.retriever.build_from_chunks(self._all_chunks)
            return True
        return False

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidate_document_ids: Optional[Sequence[str]] = None,
        retrieval_signals: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Execute BM25 keyword search and return unified SearchResult objects."""
        resp = self.retriever.retrieve(
            query=query,
            candidate_document_ids=candidate_document_ids,
            retrieval_signals=retrieval_signals,
            top_k=top_k,
        )

        results: List[SearchResult] = []
        for item in resp.results:
            meta = item.metadata or {}
            res = SearchResult(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                text=item.text,
                score=float(item.score),
                rank=item.rank,
                parent_id=meta.get("parent_id"),
                source_locator=meta.get("source_locator"),
                page=meta.get("page"),
                section=meta.get("section"),
                chunk_type=meta.get("chunk_type", "text"),
                retriever_name="keyword_bm25",
                metadata=meta,
            )
            results.append(res)

        return results

    def count(self) -> int:
        return len(self._all_chunks)
