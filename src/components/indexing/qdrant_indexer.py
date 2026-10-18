from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ..chunking.models import Chunk
from .models import SearchResult

logger = logging.getLogger(__name__)

# Sparse vector name used in Qdrant multi-vector collection
_SPARSE_VECTOR_NAME = "bm25_sparse"


def _try_load_fastembed_sparse():
    """Attempt to load fastembed BM25 sparse model. Returns model or None."""
    try:
        from fastembed import SparseTextEmbedding
        model = SparseTextEmbedding(model_name="Qdrant/bm25")
        return model
    except Exception as exc:
        logger.debug(f"fastembed sparse model unavailable: {exc}. Native hybrid search disabled.")
        return None


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


class IncompatibleCollectionError(Exception):
    """Raised when an existing collection vector dimension or distance does not match the active embedder."""
    pass


class QdrantIndexer:
    """
    Production Qdrant indexer supporting collection management,
    batch upserts, document filtering, deletion, and health checking.
    Works seamlessly in memory, local storage, or remote cluster.
    """

    def __init__(
        self,
        collection_name: str = "nexus_chunks",
        url: Optional[str] = None,
        path: Optional[str] = None,
        api_key: Optional[str] = None,
        vector_size: int = 384,
        distance: str = "Dot",
        client: Optional[QdrantClient] = None,
        create_indexes: bool = True,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 100,
        hnsw_full_scan_threshold: int = 10000,
        hnsw_on_disk: Optional[bool] = None,
        hnsw_ef_search: Optional[int] = None,
        enable_sparse: bool = False,
    ) -> None:
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance_name = distance
        self.create_indexes = create_indexes
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construct = hnsw_ef_construct
        self.hnsw_full_scan_threshold = hnsw_full_scan_threshold
        self.hnsw_on_disk = hnsw_on_disk
        self.hnsw_ef_search = hnsw_ef_search
        # Native Qdrant sparse vector (fastembed BM25) for server-side hybrid search
        self.enable_sparse = enable_sparse
        self._sparse_model = None  # Lazy-loaded on first upsert/hybrid search

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

    def _get_sparse_model(self):
        """Lazy-load fastembed BM25 sparse model on first call."""
        if not self.enable_sparse:
            return None
        if self._sparse_model is None:
            self._sparse_model = _try_load_fastembed_sparse()
            if self._sparse_model is not None:
                logger.info("Loaded fastembed BM25 sparse model for native Qdrant hybrid search.")
        return self._sparse_model

    def _encode_sparse(self, texts: List[str]) -> List[Optional[qmodels.SparseVector]]:
        """Encode texts to sparse vectors using fastembed BM25. Returns None per item on failure."""
        model = self._get_sparse_model()
        if model is None:
            return [None] * len(texts)
        try:
            sparse_vecs = list(model.embed(texts))
            results = []
            for sv in sparse_vecs:
                # fastembed SparseEmbedding has .indices and .values
                indices = [int(i) for i in sv.indices.tolist()]
                values = [float(v) for v in sv.values.tolist()]
                results.append(qmodels.SparseVector(indices=indices, values=values))
            return results
        except Exception as exc:
            logger.warning(f"Sparse encoding failed: {exc}. Falling back to dense-only search.")
            return [None] * len(texts)


    def _get_distance(self) -> qmodels.Distance:
        dist = self.distance_name.lower()
        if dist == "cosine":
            return qmodels.Distance.COSINE
        elif dist == "euclid":
            return qmodels.Distance.EUCLID
        return qmodels.Distance.DOT

    def _get_hnsw_config(self) -> qmodels.HnswConfigDiff:
        """Construct HnswConfigDiff with configured parameters."""
        return qmodels.HnswConfigDiff(
            m=self.hnsw_m,
            ef_construct=self.hnsw_ef_construct,
            full_scan_threshold=self.hnsw_full_scan_threshold,
            on_disk=self.hnsw_on_disk,
        )

    def get_hnsw_config(self) -> Dict[str, Any]:
        """Retrieve active HNSW indexing configuration for the collection."""
        try:
            coll_info = self.client.get_collection(self.collection_name)
            v_cfg = getattr(coll_info.config.params, "vectors", None)
            v_hnsw = getattr(v_cfg, "hnsw_config", None) if isinstance(v_cfg, qmodels.VectorParams) else None
            c_hnsw = getattr(coll_info.config, "hnsw_config", None)

            cfg = v_hnsw or c_hnsw
            if cfg is not None:
                return {
                    "m": getattr(cfg, "m", self.hnsw_m),
                    "ef_construct": getattr(cfg, "ef_construct", self.hnsw_ef_construct),
                    "full_scan_threshold": getattr(cfg, "full_scan_threshold", self.hnsw_full_scan_threshold),
                    "on_disk": getattr(cfg, "on_disk", self.hnsw_on_disk),
                }
        except Exception as exc:
            logger.debug(f"Failed to fetch HNSW config from collection info: {exc}")

        return {
            "m": self.hnsw_m,
            "ef_construct": self.hnsw_ef_construct,
            "full_scan_threshold": self.hnsw_full_scan_threshold,
            "on_disk": self.hnsw_on_disk,
        }

    def update_hnsw_config(
        self,
        m: Optional[int] = None,
        ef_construct: Optional[int] = None,
        full_scan_threshold: Optional[int] = None,
        on_disk: Optional[bool] = None,
    ) -> bool:
        """Dynamically update HNSW graph parameters on an existing collection."""
        if m is not None:
            self.hnsw_m = m
        if ef_construct is not None:
            self.hnsw_ef_construct = ef_construct
        if full_scan_threshold is not None:
            self.hnsw_full_scan_threshold = full_scan_threshold
        if on_disk is not None:
            self.hnsw_on_disk = on_disk

        try:
            diff = qmodels.HnswConfigDiff(
                m=self.hnsw_m,
                ef_construct=self.hnsw_ef_construct,
                full_scan_threshold=self.hnsw_full_scan_threshold,
                on_disk=self.hnsw_on_disk,
            )
            self.client.update_collection(
                collection_name=self.collection_name,
                hnsw_config=diff,
            )
            try:
                coll_info = self.client.get_collection(self.collection_name)
                v_cfg = getattr(coll_info.config.params, "vectors", None)
                if isinstance(v_cfg, qmodels.VectorParams):
                    v_cfg.hnsw_config = diff
                if hasattr(coll_info.config, "hnsw_config"):
                    coll_info.config.hnsw_config = diff
            except Exception:
                pass
            logger.info(
                f"Updated HNSW config on collection '{self.collection_name}': "
                f"m={self.hnsw_m}, ef_construct={self.hnsw_ef_construct}"
            )
            return True
        except Exception as exc:
            logger.error(f"Failed to update HNSW config on '{self.collection_name}': {exc}")
            return False

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
        """Create collection if it does not already exist, or validate existing vector configuration."""
        if getattr(self, "_collection_verified", False):
            return
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if not exists:
                hnsw_diff = self._get_hnsw_config()
                if self.enable_sparse:
                    # Create multi-vector collection: named dense + sparse for native hybrid search
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config={
                            "dense": qmodels.VectorParams(
                                size=self.vector_size,
                                distance=self._get_distance(),
                                hnsw_config=hnsw_diff,
                            )
                        },
                        sparse_vectors_config={
                            _SPARSE_VECTOR_NAME: qmodels.SparseVectorParams(
                                index=qmodels.SparseIndexParams(
                                    on_disk=False,
                                )
                            )
                        },
                        hnsw_config=hnsw_diff,
                    )
                    logger.info(
                        f"Created Qdrant collection with dense+sparse: {self.collection_name} "
                        f"(dense_size: {self.vector_size}, sparse: {_SPARSE_VECTOR_NAME})"
                    )
                else:
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=qmodels.VectorParams(
                            size=self.vector_size,
                            distance=self._get_distance(),
                            hnsw_config=hnsw_diff,
                        ),
                        hnsw_config=hnsw_diff,
                    )
                    logger.info(
                        f"Created Qdrant collection: {self.collection_name} "
                        f"(size: {self.vector_size}, HNSW: m={self.hnsw_m}, ef_construct={self.hnsw_ef_construct})"
                    )

            else:
                # Validate existing collection vector dimension
                coll_info = self.client.get_collection(self.collection_name)
                vectors_cfg = getattr(coll_info.config.params, "vectors", None)
                if isinstance(vectors_cfg, qmodels.VectorParams):
                    existing_dim = vectors_cfg.size
                    if existing_dim != self.vector_size:
                        raise IncompatibleCollectionError(
                            f"Collection '{self.collection_name}' has vector dimension {existing_dim}, "
                            f"but active embedder produces dimension {self.vector_size}. "
                            "Incompatible vector spaces cannot be mixed. "
                            "Run 'python -m src.cli ingest --rebuild-index' or call recreate_collection() to rebuild."
                        )
            if self.create_indexes:
                self.create_payload_indexes()
            self._collection_verified = True
        except Exception as exc:
            if isinstance(exc, IncompatibleCollectionError):
                raise
            logger.error(f"Error ensuring Qdrant collection: {exc}")

    def recreate_collection(self) -> None:
        """Safely delete and recreate the collection with the current vector size."""
        try:
            collections = self.client.get_collections().collections
            if any(c.name == self.collection_name for c in collections):
                self.client.delete_collection(self.collection_name)
                logger.info(f"Deleted existing Qdrant collection '{self.collection_name}'")
        except Exception as exc:
            logger.warning(f"Error deleting collection '{self.collection_name}': {exc}")

        self._collection_verified = False
        self.ensure_collection()

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[List[float]],
        batch_size: int = 100,
    ) -> int:
        """Upsert chunks with dense (and optionally sparse BM25) embeddings into Qdrant in batches."""
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

            # Encode sparse vectors for this batch if enabled
            sparse_vecs: List[Optional[qmodels.SparseVector]] = [None] * len(batch_chunks)
            if self.enable_sparse:
                batch_texts = [c.text for c in batch_chunks]
                sparse_vecs = self._encode_sparse(batch_texts)

            points: List[qmodels.PointStruct] = []
            for chunk, vec, sparse_vec in zip(batch_chunks, batch_embeddings, sparse_vecs):
                if len(vec) != self.vector_size:
                    raise ValueError(
                        f"Vector dimension mismatch for chunk '{chunk.chunk_id}': "
                        f"got {len(vec)}, expected {self.vector_size}"
                    )
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
                # Build vector: named multi-vector when sparse is enabled, plain vector otherwise
                if self.enable_sparse and sparse_vec is not None:
                    vector_payload = {
                        "dense": vec,
                        _SPARSE_VECTOR_NAME: sparse_vec,
                    }
                else:
                    vector_payload = vec  # type: ignore[assignment]

                points.append(
                    qmodels.PointStruct(
                        id=point_id,
                        vector=vector_payload,
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


    def _build_filter(
        self,
        metadata_filter: Optional[Any] = None,
        document_ids: Optional[Sequence[str]] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> Optional[qmodels.Filter]:
        """Build Qdrant filter from metadata_filter, document_ids, and legacy filter_criteria."""
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

        # 2. Explicit document scoping
        if document_ids:
            clean_ids = [str(d) for d in document_ids if str(d).strip()]
            if clean_ids:
                must_conditions.append(
                    qmodels.FieldCondition(
                        key="document_id",
                        match=qmodels.MatchAny(any=clean_ids),
                    )
                )

        # 3. Legacy dictionary filter_criteria
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

        if must_conditions or must_not_conditions:
            kwargs: Dict[str, Any] = {}
            if must_conditions:
                kwargs["must"] = must_conditions
            if must_not_conditions:
                kwargs["must_not"] = must_not_conditions
            return qmodels.Filter(**kwargs)
        return None

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        document_ids: Optional[Sequence[str]] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
        metadata_filter: Optional[Any] = None,
        hnsw_ef: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Search vector index with native metadata payload filter and optional document scoping.
        Accepts canonical RetrievalFilter, dictionary filter criteria, or explicit document IDs.
        Supports query-time HNSW search depth configuration (hnsw_ef).
        """
        q_filter = self._build_filter(metadata_filter, document_ids, filter_criteria)

        # Configure HNSW search parameters if specified or configured
        effective_ef = hnsw_ef if hnsw_ef is not None else self.hnsw_ef_search
        search_params = None
        if effective_ef is not None:
            search_params = qmodels.SearchParams(hnsw_ef=effective_ef)

        if hasattr(self.client, "query_points"):
            qp_kwargs: Dict[str, Any] = {
                "collection_name": self.collection_name,
                "query": query_vector,
                "query_filter": q_filter,
                "limit": top_k,
            }
            if self.enable_sparse:
                qp_kwargs["using"] = "dense"
            if search_params is not None:
                qp_kwargs["search_params"] = search_params
            response = self.client.query_points(**qp_kwargs)
            results = response.points
        elif hasattr(self.client, "search"):
            s_kwargs: Dict[str, Any] = {
                "collection_name": self.collection_name,
                "query_vector": ("dense", query_vector) if self.enable_sparse else query_vector,
                "query_filter": q_filter,
                "limit": top_k,
            }
            if search_params is not None:
                s_kwargs["search_params"] = search_params
            results = self.client.search(**s_kwargs)
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

    def hybrid_search(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: int = 10,
        document_ids: Optional[Sequence[str]] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
        metadata_filter: Optional[Any] = None,
        hnsw_ef: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Qdrant Native Hybrid Search combining dense semantic search and fastembed BM25 sparse search
        via server-side Reciprocal Rank Fusion (Prefetch + FusionQuery(Rrf)).
        Provides single-engine transactional updates and unified payload filtering inside Qdrant Rust engine.
        """
        if not self.enable_sparse or not hasattr(self.client, "query_points"):
            return self.search(
                query_vector=query_vector,
                top_k=top_k,
                document_ids=document_ids,
                filter_criteria=filter_criteria,
                metadata_filter=metadata_filter,
                hnsw_ef=hnsw_ef,
            )

        sparse_encoded = self._encode_sparse([query_text])
        sparse_vec = sparse_encoded[0] if sparse_encoded else None

        if sparse_vec is None:
            return self.search(
                query_vector=query_vector,
                top_k=top_k,
                document_ids=document_ids,
                filter_criteria=filter_criteria,
                metadata_filter=metadata_filter,
                hnsw_ef=hnsw_ef,
            )

        q_filter = self._build_filter(metadata_filter, document_ids, filter_criteria)
        prefetch_limit = max(top_k * 2, 20)

        prefetches = [
            qmodels.Prefetch(
                query=query_vector,
                using="dense",
                filter=q_filter,
                limit=prefetch_limit,
            ),
            qmodels.Prefetch(
                query=sparse_vec,
                using=_SPARSE_VECTOR_NAME,
                filter=q_filter,
                limit=prefetch_limit,
            ),
        ]

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetches,
                query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
                limit=top_k,
            )
            results = response.points
        except Exception as exc:
            logger.warning(f"Native Qdrant hybrid search failed: {exc}. Falling back to dense search.")
            return self.search(
                query_vector=query_vector,
                top_k=top_k,
                document_ids=document_ids,
                filter_criteria=filter_criteria,
                metadata_filter=metadata_filter,
                hnsw_ef=hnsw_ef,
            )

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
                retriever_name="native_hybrid_qdrant",
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
