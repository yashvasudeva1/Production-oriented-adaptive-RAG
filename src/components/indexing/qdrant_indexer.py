from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ..chunking.models import Chunk
from .models import SearchResult

logger = logging.getLogger(__name__)


DEFAULT_PAYLOAD_INDEXES: Dict[str, qmodels.PayloadSchemaType] = {
    "document_id": qmodels.PayloadSchemaType.KEYWORD,
    "chunk_id": qmodels.PayloadSchemaType.KEYWORD,
    "chunk_type": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.document_type": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.category": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.year": qmodels.PayloadSchemaType.INTEGER,
    "metadata.date": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.dates": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.organization": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.organizations": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.department": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.location": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.locations": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.language": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.access": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.security": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.source": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.tenant_id": qmodels.PayloadSchemaType.KEYWORD,
}


class QdrantIndexer:
    """
    Production Qdrant indexer supporting collection management,
    batch upserts, document filtering, deletion, and health checking.
    Works seamlessly in memory, local storage, or remote cluster.
    """

    def __init__(
        self,
        collection_name: str = "researchlens_chunks",
        url: Optional[str] = None,
        path: Optional[str] = None,
        api_key: Optional[str] = None,
        vector_size: int = 384,
        distance: str = "Dot",
        client: Optional[QdrantClient] = None,
        create_indexes: bool = True,
    ) -> None:
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance_name = distance
        self.create_indexes = create_indexes

        if client is not None:
            self.client = client
        elif url:
            self.client = QdrantClient(url=url, api_key=api_key)
        elif path:
            self.client = QdrantClient(path=path)
        else:
            # Default to in-memory mode for fast zero-dependency local execution
            self.client = QdrantClient(":memory:")

        self.ensure_collection()

    def _get_distance(self) -> qmodels.Distance:
        dist = self.distance_name.lower()
        if dist == "cosine":
            return qmodels.Distance.COSINE
        elif dist == "euclid":
            return qmodels.Distance.EUCLID
        return qmodels.Distance.DOT

    def create_payload_indexes(
        self, indexes: Optional[Dict[str, qmodels.PayloadSchemaType]] = None
    ) -> List[str]:
        """
        Create payload indexes for frequently filtered fields.
        Idempotent, logged, safe if already created, and handles memory clients gracefully.
        """
        targets = indexes or DEFAULT_PAYLOAD_INDEXES
        indexed_fields: List[str] = []

        for field_name, schema_type in targets.items():
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
                indexed_fields.append(field_name)
                logger.debug(f"Created payload index for '{field_name}' ({schema_type})")
            except Exception as exc:
                # In-memory Qdrant or already existing index returns ok/warning
                msg = str(exc).lower()
                if "already exists" in msg or "already indexed" in msg:
                    indexed_fields.append(field_name)
                else:
                    logger.debug(f"Payload index notice for '{field_name}': {exc}")
                    indexed_fields.append(field_name)

        logger.info(
            f"Configured {len(indexed_fields)} payload indexes on collection '{self.collection_name}'"
        )
        return indexed_fields

    def ensure_collection(self) -> None:
        """Create collection if it does not already exist and initialize payload indexes."""
        if getattr(self, "_collection_verified", False):
            return
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.vector_size,
                        distance=self._get_distance(),
                    ),
                )
                logger.info(
                    f"Created Qdrant collection: {self.collection_name} (size: {self.vector_size})"
                )
            if self.create_indexes:
                self.create_payload_indexes()
            self._collection_verified = True
        except Exception as exc:
            logger.error(f"Error ensuring Qdrant collection: {exc}")

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[List[float]],
        batch_size: int = 100,
    ) -> int:
        """Upsert chunks with embeddings into Qdrant in batches."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
            )

        if not chunks:
            return 0

        total_upserted = 0
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            points: List[qmodels.PointStruct] = []
            for chunk, vec in zip(batch_chunks, batch_embeddings):
                # Deterministic point UUID from chunk_id
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
                payload = {
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
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
                points.append(
                    qmodels.PointStruct(
                        id=point_id,
                        vector=vec,
                        payload=payload,
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            total_upserted += len(points)

        if total_upserted > 0:
            try:
                from ..cache import CacheManager
                CacheManager.get_instance().bump_index_version()
            except Exception:
                pass

        logger.info(f"Upserted {total_upserted} chunks into Qdrant '{self.collection_name}'")
        return total_upserted

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        document_ids: Optional[Sequence[str]] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
        metadata_filter: Optional[Any] = None,
    ) -> List[SearchResult]:
        """
        Search vector index with native metadata payload filter and optional document scoping.
        Accepts canonical RetrievalFilter, dictionary filter criteria, or explicit document IDs.
        """
        must_conditions: List[qmodels.Condition] = []
        must_not_conditions: List[qmodels.Condition] = []

        # 1. Native RetrievalFilter processing
        if metadata_filter is not None:
            if hasattr(metadata_filter, "to_qdrant_filter"):
                q_filt = metadata_filter.to_qdrant_filter()
            elif isinstance(metadata_filter, dict):
                from ..retrieval.filters import RetrievalFilter
                q_filt = RetrievalFilter.from_dict(metadata_filter).to_qdrant_filter()
            else:
                q_filt = None

            if q_filt is not None:
                if getattr(q_filt, "must", None):
                    must_conditions.extend(q_filt.must)
                if getattr(q_filt, "must_not", None):
                    must_not_conditions.extend(q_filt.must_not)

        # 2. Explicit document scoping (if specified separately)
        if document_ids:
            clean_ids = [str(d) for d in document_ids if str(d).strip()]
            if clean_ids:
                must_conditions.append(
                    qmodels.FieldCondition(
                        key="document_id",
                        match=qmodels.MatchAny(any=clean_ids),
                    )
                )

        # 3. Legacy dictionary filter_criteria (if specified separately)
        if filter_criteria:
            for key, val in filter_criteria.items():
                if val is not None:
                    target_key = key if key.startswith("metadata.") or key in ("document_id", "chunk_id", "chunk_type") else f"metadata.{key}"
                    if isinstance(val, (list, tuple, set)):
                        clean_vals = [str(v) for v in val if v is not None]
                        must_conditions.append(
                            qmodels.FieldCondition(
                                key=target_key,
                                match=qmodels.MatchAny(any=clean_vals),
                            )
                        )
                    else:
                        must_conditions.append(
                            qmodels.FieldCondition(
                                key=target_key,
                                match=qmodels.MatchValue(value=val),
                            )
                        )

        q_filter = None
        if must_conditions or must_not_conditions:
            kwargs: Dict[str, Any] = {}
            if must_conditions:
                kwargs["must"] = must_conditions
            if must_not_conditions:
                kwargs["must_not"] = must_not_conditions
            q_filter = qmodels.Filter(**kwargs)

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=q_filter,
                limit=top_k,
            )
            results = response.points
        elif hasattr(self.client, "search"):
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=q_filter,
                limit=top_k,
            )
        else:
            results = []

        search_results: List[SearchResult] = []
        for rank, hit in enumerate(results, start=1):
            payload = hit.payload or {}
            res = SearchResult(
                chunk_id=payload.get("chunk_id", ""),
                document_id=payload.get("document_id", ""),
                text=payload.get("text", ""),
                score=float(hit.score),
                rank=rank,
                parent_id=payload.get("parent_id"),
                source_locator=payload.get("source_locator"),
                page=payload.get("page"),
                section=payload.get("section"),
                chunk_type=payload.get("chunk_type", "text"),
                retriever_name="dense_qdrant",
                metadata=payload.get("metadata", {}),
            )
            search_results.append(res)

        return search_results

    def delete_by_document_id(self, document_id: str) -> bool:
        """Delete all points associated with a specific document ID."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=str(document_id)),
                        )
                    ]
                ),
            )
            logger.info(f"Deleted document {document_id} from Qdrant")
            try:
                from ..cache import CacheManager
                CacheManager.get_instance().bump_index_version()
            except Exception:
                pass
            return True
        except Exception as exc:
            logger.error(f"Failed to delete document {document_id} from Qdrant: {exc}")
            return False

    def health_check(self) -> bool:
        """Check if Qdrant is connected and the collection exists."""
        try:
            collections = self.client.get_collections().collections
            return any(c.name == self.collection_name for c in collections)
        except Exception:
            return False

    def count(self) -> int:
        """Return total chunks currently indexed."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0
