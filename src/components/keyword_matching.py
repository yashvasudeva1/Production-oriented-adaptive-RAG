from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from dotenv import load_dotenv
load_dotenv()
DEFAULT_CHUNKS_PATH = (
    Path(__file__).resolve().parents[2] / "metadata" / "chunks.json"
)


# Models

class BM25Result(BaseModel):
    """One ranked BM25 retrieval result."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class BM25RetrievalResponse(BaseModel):
    """Structured output returned by the retriever."""

    query: str
    effective_query: str

    candidate_chunks: int
    returned_chunks: int

    candidate_document_ids: list[str] = Field(default_factory=list)
    results: list[BM25Result] = Field(default_factory=list)


@dataclass
class BM25Config:
    """Configuration for the BM25 retriever."""

    chunks_path: str | Path = DEFAULT_CHUNKS_PATH

    default_top_k: int = 5
    max_top_k: int = 100

    # The component supports common chunk schemas.
    text_fields: tuple[str, ...] = (
        "text",
        "content",
        "page_content",
    )

    # BM25 parameters.
    k1: float = 1.5
    b: float = 0.75

    lowercase: bool = True
    remove_punctuation: bool = True
    min_token_length: int = 1

    # Keep empty by default. Over-aggressive stopword removal can hurt factual
    # retrieval because terms such as "not", "without", etc. can matter.
    stopwords: set[str] = field(default_factory=set)


# Text processing

def normalize_text(
    text: str,
    *,
    lowercase: bool = True,
    remove_punctuation: bool = True,
) -> str:
    """Normalize text while preserving useful lexical information."""

    text = str(text or "").strip()

    if lowercase:
        text = text.lower()

    if remove_punctuation:
        # Convert punctuation to spaces instead of deleting it outright.
        text = re.sub(r"[^a-z0-9\u0080-\uffff]+", " ", text)

    return re.sub(r"\s+", " ", text).strip()


def tokenize(
    text: str,
    *,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    min_token_length: int = 1,
    stopwords: set[str] | None = None,
) -> list[str]:
    """Convert text into tokens for BM25."""

    normalized = normalize_text(
        text,
        lowercase=lowercase,
        remove_punctuation=remove_punctuation,
    )

    tokens = normalized.split()

    if min_token_length > 1:
        tokens = [
            token
            for token in tokens
            if len(token) >= min_token_length
        ]

    if stopwords:
        normalized_stopwords = {
            normalize_text(word)
            for word in stopwords
            if normalize_text(word)
        }

        tokens = [
            token
            for token in tokens
            if token not in normalized_stopwords
        ]

    return tokens


# Chunk loading

def _load_chunk_records(path: str | Path) -> list[dict[str, Any]]:
    """
    Load chunk records from JSON.

    Supported formats:

    [
        {"chunk_id": "...", "document_id": "...", "text": "..."}
    ]

    or:

    {"chunks": [...]}
    """

    chunk_path = Path(path)

    if not chunk_path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {chunk_path}"
        )

    with chunk_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        records = data

    elif isinstance(data, Mapping):
        records = None

        for key in ("chunks", "documents", "records", "items"):
            if isinstance(data.get(key), list):
                records = data[key]
                break

        if records is None:
            raise ValueError(
                "Chunk JSON must be a list or contain a list under "
                "'chunks', 'documents', 'records', or 'items'."
            )
    else:
        raise ValueError("Invalid chunk JSON format.")

    valid_records: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue

        document_id = (
            record.get("document_id")
            or record.get("doc_id")
            or record.get("parent_document_id")
        )

        chunk_id = (
            record.get("chunk_id")
            or record.get("id")
            or f"chunk-{index}"
        )

        if not document_id:
            continue

        valid_records.append(
            {
                **dict(record),
                "document_id": str(document_id),
                "chunk_id": str(chunk_id),
            }
        )

    return valid_records


def _extract_text(
    record: Mapping[str, Any],
    text_fields: Sequence[str],
) -> str:
    """Get searchable text from a chunk record."""

    for field_name in text_fields:
        value = record.get(field_name)

        if value is None:
            continue

        text = str(value).strip()

        if text:
            return text

    return ""


# Query handling

def _build_effective_query(
    query: str,
    retrieval_signals: Mapping[str, Any] | None = None,
) -> str:
    """
    Build the lexical query.

    The original user query is always the primary signal. Optional topics
    extracted by the metadata stage are appended only when they add new terms.
    """

    query = str(query or "").strip()

    if not retrieval_signals:
        return query

    topics = retrieval_signals.get("topics", [])

    if isinstance(topics, str):
        topics = [topics]

    existing = normalize_text(query)
    additions: list[str] = []

    for topic in topics or []:
        topic = str(topic).strip()

        if not topic:
            continue

        normalized_topic = normalize_text(topic)

        if normalized_topic and normalized_topic not in existing:
            additions.append(topic)

    return f"{query} {' '.join(additions)}".strip()


# BM25 Retriever

class BM25Retriever:
    """
    General-purpose BM25 keyword retriever.

    Responsibility:
        Query + optional candidate documents -> ranked chunks

    It intentionally does NOT:
        - classify queries
        - extract query metadata
        - perform metadata filtering
        - rerank results
        - generate answers
    """

    def __init__(
        self,
        config: BM25Config | None = None,
    ) -> None:
        self.config = config or BM25Config()

        self._chunks: list[dict[str, Any]] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

        self.reload()

    # Index management

    def reload(self) -> None:
        """Reload chunks from disk and rebuild the BM25 index."""

        chunks = _load_chunk_records(
            self.config.chunks_path
        )

        self.build_from_chunks(chunks)

    def build_from_chunks(
        self,
        chunks: Sequence[Mapping[str, Any]],
    ) -> None:
        """
        Build the BM25 index directly from in-memory chunks.

        This is useful when a separate chunking pipeline produces chunks.
        """

        valid_chunks: list[dict[str, Any]] = []
        tokenized_corpus: list[list[str]] = []

        for index, chunk in enumerate(chunks):
            document_id = (
                chunk.get("document_id")
                or chunk.get("doc_id")
                or chunk.get("parent_document_id")
            )

            if not document_id:
                continue

            chunk_id = (
                chunk.get("chunk_id")
                or chunk.get("id")
                or f"chunk-{index}"
            )

            text = _extract_text(
                chunk,
                self.config.text_fields,
            )

            if not text:
                continue

            tokens = tokenize(
                text,
                lowercase=self.config.lowercase,
                remove_punctuation=self.config.remove_punctuation,
                min_token_length=self.config.min_token_length,
                stopwords=self.config.stopwords,
            )

            if not tokens:
                continue

            valid_chunks.append(
                {
                    **dict(chunk),
                    "document_id": str(document_id),
                    "chunk_id": str(chunk_id),
                }
            )

            tokenized_corpus.append(tokens)

        self._chunks = valid_chunks
        self._tokenized_corpus = tokenized_corpus

        if not tokenized_corpus:
            self._bm25 = None
            return

        self._bm25 = BM25Okapi(
            tokenized_corpus,
            k1=self.config.k1,
            b=self.config.b,
        )

    # Candidate selection

    def _candidate_indices(
        self,
        candidate_document_ids: Sequence[str] | None,
    ) -> list[int]:
        """Return chunk indices belonging to candidate documents."""

        if not candidate_document_ids:
            return list(range(len(self._chunks)))

        candidate_set = {
            str(document_id)
            for document_id in candidate_document_ids
            if str(document_id).strip()
        }

        return [
            index
            for index, chunk in enumerate(self._chunks)
            if chunk["document_id"] in candidate_set
        ]

    # Retrieval

    def retrieve(
        self,
        query: str,
        *,
        candidate_document_ids: Sequence[str] | None = None,
        retrieval_signals: Mapping[str, Any] | None = None,
        top_k: int | None = None,
    ) -> BM25RetrievalResponse:
        """
        Retrieve the highest-scoring lexical matches.

        Parameters
        ----------
        query:
            Original user query.

        candidate_document_ids:
            Document IDs returned by MetadataFilter. When omitted, the full
            chunk corpus is searched.

        retrieval_signals:
            Optional output from MetadataFilter, e.g.
            {"topics": ["Transformer"]}.

        top_k:
            Number of chunks to return.
        """

        query = str(query or "").strip()

        requested_top_k = (
            self.config.default_top_k
            if top_k is None
            else int(top_k)
        )

        requested_top_k = max(
            1,
            min(requested_top_k, self.config.max_top_k),
        )

        effective_query = _build_effective_query(
            query,
            retrieval_signals,
        )

        if not query:
            return BM25RetrievalResponse(
                query=query,
                effective_query=effective_query,
                candidate_chunks=0,
                returned_chunks=0,
                candidate_document_ids=list(
                    candidate_document_ids or []
                ),
                results=[],
            )

        if self._bm25 is None:
            return BM25RetrievalResponse(
                query=query,
                effective_query=effective_query,
                candidate_chunks=0,
                returned_chunks=0,
                candidate_document_ids=list(
                    candidate_document_ids or []
                ),
                results=[],
            )

        query_tokens = tokenize(
            effective_query,
            lowercase=self.config.lowercase,
            remove_punctuation=self.config.remove_punctuation,
            min_token_length=self.config.min_token_length,
            stopwords=self.config.stopwords,
        )

        if not query_tokens:
            return BM25RetrievalResponse(
                query=query,
                effective_query=effective_query,
                candidate_chunks=0,
                returned_chunks=0,
                candidate_document_ids=list(
                    candidate_document_ids or []
                ),
                results=[],
            )

        candidate_indices = self._candidate_indices(
            candidate_document_ids
        )

        if not candidate_indices:
            return BM25RetrievalResponse(
                query=query,
                effective_query=effective_query,
                candidate_chunks=0,
                returned_chunks=0,
                candidate_document_ids=list(
                    candidate_document_ids or []
                ),
                results=[],
            )

        # rank_bm25 scores against the complete indexed corpus.
        # We then restrict ranking to the candidate document set produced by
        # metadata filtering.
        scores = self._bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            candidate_indices,
            key=lambda index: float(scores[index]),
            reverse=True,
        )

        # Prefer actual lexical matches. If none have positive scores, return
        # the highest-scoring candidate chunks anyway so the caller can decide
        # how to handle a lexical miss.
        positive_indices = [
            index
            for index in ranked_indices
            if float(scores[index]) > 0
        ]

        selected_indices = (
            positive_indices[:requested_top_k]
            if positive_indices
            else ranked_indices[:requested_top_k]
        )

        results: list[BM25Result] = []

        for rank, index in enumerate(selected_indices, start=1):
            chunk = self._chunks[index]

            text = _extract_text(
                chunk,
                self.config.text_fields,
            )

            metadata = dict(chunk)

            for field_name in self.config.text_fields:
                metadata.pop(field_name, None)

            results.append(
                BM25Result(
                    chunk_id=chunk["chunk_id"],
                    document_id=chunk["document_id"],
                    text=text,
                    score=float(scores[index]),
                    rank=rank,
                    metadata=metadata,
                )
            )

        return BM25RetrievalResponse(
            query=query,
            effective_query=effective_query,
            candidate_chunks=len(candidate_indices),
            returned_chunks=len(results),
            candidate_document_ids=list(
                candidate_document_ids or []
            ),
            results=results,
        )

    # Metadata-filter integration

    def retrieve_from_filter_result(
        self,
        query: str,
        filter_result: Any,
        *,
        top_k: int | None = None,
    ) -> BM25RetrievalResponse:
        """
        Run BM25 directly from the output of MetadataFilter.

        Supports both:
            FilterResult instances
        and:
            dictionaries produced by model_dump().
        """

        if hasattr(filter_result, "document_ids"):
            document_ids = filter_result.document_ids
            retrieval_signals = filter_result.retrieval_signals

        elif isinstance(filter_result, Mapping):
            document_ids = filter_result.get(
                "document_ids",
                [],
            )
            retrieval_signals = filter_result.get(
                "retrieval_signals",
                {},
            )

        else:
            raise TypeError(
                "filter_result must be a FilterResult-like object "
                "or a mapping."
            )

        return self.retrieve(
            query,
            candidate_document_ids=document_ids,
            retrieval_signals=retrieval_signals,
            top_k=top_k,
        )


# Functional API

def retrieve_bm25(
    query: str,
    *,
    candidate_document_ids: Sequence[str] | None = None,
    retrieval_signals: Mapping[str, Any] | None = None,
    top_k: int = 5,
    chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
) -> BM25RetrievalResponse:
    """Convenience function for one-off BM25 retrieval."""

    retriever = BM25Retriever(
        BM25Config(
            chunks_path=chunks_path,
        )
    )

    return retriever.retrieve(
        query,
        candidate_document_ids=candidate_document_ids,
        retrieval_signals=retrieval_signals,
        top_k=top_k,
    )


# Example

if __name__ == "__main__":
    """
    Expected metadata/chunks.json structure:

    [
        {
            "chunk_id": "chunk-001",
            "document_id": "document-001",
            "text": "The Transformer was proposed in 2017..."
        },
        {
            "chunk_id": "chunk-002",
            "document_id": "document-001",
            "text": "The architecture relies entirely on attention..."
        }
    ]

    Run from the project root:

        python -m src.components.bm25_retriever
    """

    retriever = BM25Retriever()

    query = "What did the Transformer paper propose?"

    response = retriever.retrieve(
        query,
        top_k=5,
    )

    print("-" * 70)
    print("QUERY")
    print("-" * 70)
    print(query)

    print("\n" + "-" * 70)
    print("BM25 RESULTS")
    print("-" * 70)

    for result in response.results:
        print(
            f"\nRank: {result.rank}"
            f"\nScore: {result.score:.4f}"
            f"\nChunk: {result.chunk_id}"
            f"\nDocument: {result.document_id}"
            f"\nText: {result.text[:500]}"
        )
