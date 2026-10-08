from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Union
from pydantic import BaseModel, Field

try:
    from qdrant_client.http import models as qmodels
except ImportError:
    qmodels = None  # type: ignore

logger = logging.getLogger(__name__)

FilterOperator = Literal[
    "eq", "ne", "in", "nin", "gt", "gte", "lt", "lte", "contains"
]


class FilterCondition(BaseModel):
    """
    Structured atomic filter constraint.
    Example: field='metadata.year', operator='eq', value=2025
    """
    field: str
    operator: FilterOperator = "eq"
    value: Any

    def get_normalized_key(self) -> str:
        """Ensure field has clean naming, prepending 'metadata.' if document-level property."""
        k = self.field.strip()
        top_level_keys = {"document_id", "chunk_id", "chunk_type", "page", "section", "source_locator"}
        if k in top_level_keys:
            return k
        if not k.startswith("metadata."):
            return f"metadata.{k}"
        return k

    def get_raw_field_name(self) -> str:
        """Strip 'metadata.' prefix if present."""
        k = self.field.strip()
        if k.startswith("metadata."):
            return k[len("metadata."):]
        return k


class SoftPreference(BaseModel):
    """
    Soft ranking preference that influences score rather than strictly eliminating documents.
    Example: field='metadata.year', preferred_value=2025, boost_weight=0.1
    """
    field: str
    preferred_value: Any
    boost_weight: float = 0.1


class RetrievalHints(BaseModel):
    """
    Semantic and syntactic hints to steer retrieval strategy (e.g. boost exact BM25 matches).
    """
    exact_terms: List[str] = Field(default_factory=list)
    quoted_phrases: List[str] = Field(default_factory=list)
    technical_identifiers: List[str] = Field(default_factory=list)
    numeric_values: List[str] = Field(default_factory=list)
    code_symbols: List[str] = Field(default_factory=list)
    semantic_intent: Optional[str] = None


class RetrievalFilter(BaseModel):
    """
    Canonical structured retrieval filter abstraction.
    Enforces identical hard constraints natively across Dense (Qdrant) and Lexical (BM25) search.
    """
    hard_filters: List[FilterCondition] = Field(default_factory=list)
    soft_preferences: List[SoftPreference] = Field(default_factory=list)
    retrieval_hints: RetrievalHints = Field(default_factory=RetrievalHints)
    document_ids: Optional[List[str]] = Field(default=None)

    def is_empty(self) -> bool:
        return not self.hard_filters and not self.document_ids

    def add_hard_filter(self, field: str, value: Any, operator: FilterOperator = "eq") -> RetrievalFilter:
        self.hard_filters.append(FilterCondition(field=field, operator=operator, value=value))
        return self

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> RetrievalFilter:
        """Construct RetrievalFilter from dictionary (supporting legacy and structured payloads)."""
        if not data:
            return cls()

        # If already formatted as a full RetrievalFilter dict
        if "hard_filters" in data or "retrieval_hints" in data:
            hard = [
                FilterCondition(**hf) if isinstance(hf, dict) else hf
                for hf in data.get("hard_filters", [])
            ]
            soft = [
                SoftPreference(**sp) if isinstance(sp, dict) else sp
                for sp in data.get("soft_preferences", [])
            ]
            hints_raw = data.get("retrieval_hints", {})
            hints = hints_raw if isinstance(hints_raw, RetrievalHints) else RetrievalHints(**hints_raw)
            return cls(
                hard_filters=hard,
                soft_preferences=soft,
                retrieval_hints=hints,
                document_ids=data.get("document_ids"),
            )

        # Legacy flat dictionary conversion (e.g. {"document_type": "pdf", "year": 2024})
        rf = cls()
        for k, v in data.items():
            if v is None:
                continue
            if k == "document_ids":
                rf.document_ids = [str(x) for x in v if str(x).strip()]
            elif k in ("exact_identifiers", "topics", "quoted_phrases"):
                if k == "exact_identifiers" and isinstance(v, list):
                    rf.retrieval_hints.technical_identifiers.extend(v)
                elif k == "quoted_phrases" and isinstance(v, list):
                    rf.retrieval_hints.quoted_phrases.extend(v)
            elif isinstance(v, (list, tuple, set)):
                clean_v = [x for x in v if x is not None and str(x).strip()]
                if clean_v:
                    rf.hard_filters.append(
                        FilterCondition(field=k, operator="in", value=clean_v)
                    )
            else:
                rf.hard_filters.append(
                    FilterCondition(field=k, operator="eq", value=v)
                )
        return rf

    def to_qdrant_filter(self) -> Optional[Any]:
        """
        Convert canonical hard filters and document IDs into native Qdrant payload filters.
        Avoids huge candidate-document-ID lists by directly targeting payload fields.
        """
        if qmodels is None:
            return None

        must_conditions: List[Any] = []
        must_not_conditions: List[Any] = []

        # Explicit document scoping (if specified by caller)
        if self.document_ids:
            clean_ids = [str(d).strip() for d in self.document_ids if str(d).strip()]
            if clean_ids:
                must_conditions.append(
                    qmodels.FieldCondition(
                        key="document_id",
                        match=qmodels.MatchAny(any=clean_ids),
                    )
                )

        for cond in self.hard_filters:
            key = cond.get_normalized_key()
            op = cond.operator
            val = cond.value

            if val is None:
                continue

            if op == "eq":
                if isinstance(val, (int, float, bool)):
                    must_conditions.append(
                        qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=val))
                    )
                else:
                    must_conditions.append(
                        qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=str(val)))
                    )
            elif op == "ne":
                must_not_conditions.append(
                    qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=val))
                )
            elif op == "in":
                items = list(val) if isinstance(val, (list, tuple, set)) else [val]
                if items:
                    must_conditions.append(
                        qmodels.FieldCondition(key=key, match=qmodels.MatchAny(any=items))
                    )
            elif op == "nin":
                items = list(val) if isinstance(val, (list, tuple, set)) else [val]
                if items:
                    must_not_conditions.append(
                        qmodels.FieldCondition(key=key, match=qmodels.MatchAny(any=items))
                    )
            elif op in ("gt", "gte", "lt", "lte"):
                range_kwargs: Dict[str, Any] = {}
                if op == "gt":
                    range_kwargs["gt"] = float(val)
                elif op == "gte":
                    range_kwargs["gte"] = float(val)
                elif op == "lt":
                    range_kwargs["lt"] = float(val)
                elif op == "lte":
                    range_kwargs["lte"] = float(val)
                must_conditions.append(
                    qmodels.FieldCondition(key=key, range=qmodels.Range(**range_kwargs))
                )
            elif op == "contains":
                must_conditions.append(
                    qmodels.FieldCondition(key=key, match=qmodels.MatchText(text=str(val)))
                )

        if not must_conditions and not must_not_conditions:
            return None

        kwargs: Dict[str, Any] = {}
        if must_conditions:
            kwargs["must"] = must_conditions
        if must_not_conditions:
            kwargs["must_not"] = must_not_conditions

        return qmodels.Filter(**kwargs)

    def matches_chunk(self, chunk: Any) -> bool:
        """
        Evaluate hard filters in-memory for lexical (BM25) search or verification.
        Ensures dense and lexical branches evaluate the exact same candidate universe.
        """
        # 1. Document ID scoping
        if self.document_ids:
            clean_ids = {str(d).strip() for d in self.document_ids if str(d).strip()}
            doc_id = str(_extract_val(chunk, "document_id") or "")
            if doc_id not in clean_ids:
                return False

        # 2. Hard filter conditions
        for cond in self.hard_filters:
            raw_field = cond.get_raw_field_name()
            actual_val = _extract_val(chunk, raw_field)
            if actual_val is None:
                # Also try top level if raw_field wasn't prefixed
                actual_val = _extract_val(chunk, cond.field)

            if not _evaluate_operator(actual_val, cond.operator, cond.value):
                return False

        return True


def _extract_val(obj: Any, field_name: str) -> Any:
    """Extract field from dictionary, Chunk model, or nested metadata."""
    if isinstance(obj, dict):
        if field_name in obj:
            return obj[field_name]
        meta = obj.get("metadata")
        if isinstance(meta, dict) and field_name in meta:
            return meta[field_name]
        # Check if field_name starts with metadata.
        if field_name.startswith("metadata."):
            sub_k = field_name[len("metadata."):]
            if isinstance(meta, dict) and sub_k in meta:
                return meta[sub_k]
        return None

    # Object / Chunk dataclass / Pydantic model
    val = getattr(obj, field_name, None)
    if val is not None:
        return val
    meta = getattr(obj, "metadata", None)
    if isinstance(meta, dict) and field_name in meta:
        return meta[field_name]
    if field_name.startswith("metadata."):
        sub_k = field_name[len("metadata."):]
        if isinstance(meta, dict) and sub_k in meta:
            return meta[sub_k]
    return None


def _evaluate_operator(actual: Any, op: FilterOperator, expected: Any) -> bool:
    if actual is None:
        return False

    def _norm(x: Any) -> str:
        return str(x).strip().lower()

    if op == "eq":
        if isinstance(actual, (list, tuple, set)):
            exp_str = _norm(expected)
            return any(_norm(item) == exp_str for item in actual)
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float, str)):
            try:
                return float(actual) == float(expected)
            except (ValueError, TypeError):
                pass
        return _norm(actual) == _norm(expected)

    elif op == "ne":
        return not _evaluate_operator(actual, "eq", expected)

    elif op == "in":
        expected_set = {_norm(x) for x in expected} if isinstance(expected, (list, tuple, set)) else {_norm(expected)}
        if isinstance(actual, (list, tuple, set)):
            return any(_norm(item) in expected_set for item in actual)
        return _norm(actual) in expected_set

    elif op == "nin":
        return not _evaluate_operator(actual, "in", expected)

    elif op in ("gt", "gte", "lt", "lte"):
        try:
            num_actual = float(actual)
            num_expected = float(expected)
            if op == "gt":
                return num_actual > num_expected
            elif op == "gte":
                return num_actual >= num_expected
            elif op == "lt":
                return num_actual < num_expected
            elif op == "lte":
                return num_actual <= num_expected
        except (ValueError, TypeError):
            return False

    elif op == "contains":
        return _norm(expected) in _norm(actual)

    return True


# Backward-compatible alias
QueryConstraints = RetrievalFilter
