from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

from ..query_metadata_extraction import QueryMetadata, QueryMetadataExtractor

logger = logging.getLogger(__name__)

# Phrases that indicate the user is explicitly requesting date-scoped documents
_DATE_FILTER_PHRASES = (
    "documents from",
    "papers from",
    "published in",
    "dated",
    "in the year",
    "from the year",
    "filed in",
    "issued in",
    "show me",
    "find me",
    "retrieve",
    "documents dated",
    "from 20",
    "from 19",
)


class AdaptiveQueryMetadataExtractor:
    """
    Extracts explicit metadata constraints from user queries.
    Uses open source models via Groq or Hugging Face, with
    deterministic regex-based extraction as a fallback.

    Each extracted field carries a '_confidence' sub-key indicating whether
    the field was explicitly requested ('high'), heuristically inferred
    ('medium'), or weakly implied ('low').  The QueryPlanner uses this to
    decide whether the field becomes a hard filter or a soft ranking hint.
    """

    def __init__(self, use_llm_if_available: bool = True) -> None:
        self.use_llm = use_llm_if_available
        self._groq_extractor: Optional[Any] = None
        self._hf_extractor: Optional[QueryMetadataExtractor] = None

        if self.use_llm and os.getenv("GROQ_API_KEY") and not os.getenv("OFFLINE_EVAL"):
            try:
                from langchain_groq import ChatGroq
                model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
                self._groq_extractor = ChatGroq(
                    model=model_name,
                    api_key=os.getenv("GROQ_API_KEY"),
                    temperature=0.0,
                    max_retries=1,
                    request_timeout=5,
                ).with_structured_output(QueryMetadata)
            except Exception as exc:
                logger.warning(f"Could not initialize Groq metadata extractor: {exc}")

        if self.use_llm and os.getenv("HF_TOKEN") and not self._groq_extractor and not os.getenv("OFFLINE_EVAL"):
            try:
                self._hf_extractor = QueryMetadataExtractor()
            except Exception as exc:
                logger.warning(f"Could not initialize HF metadata extractor: {exc}")

    def extract(self, query: str) -> Dict[str, Any]:
        # Fast sub-millisecond heuristic check first
        rule_extracted = self._extract_rule_based(query)
        if (
            rule_extracted["document_type"]
            or rule_extracted["organizations"]
            or rule_extracted["dates"]
            or os.getenv("OFFLINE_EVAL")
        ):
            return rule_extracted

        if self._groq_extractor:
            try:
                import json
                from pathlib import Path
                prompt_file = Path(__file__).resolve().parents[3] / "prompts" / "query_metadata_extraction.json"
                system_prompt = "Extract explicit metadata constraints from the query. Return only JSON matching schema."
                if prompt_file.exists():
                    data = json.loads(prompt_file.read_text(encoding="utf-8"))
                    system_prompt = data.get("query_metadata_extraction", {}).get("system", system_prompt)
                extracted = self._groq_extractor.invoke([
                    ("system", system_prompt),
                    ("human", query),
                ])
                # LLM-extracted fields are HIGH confidence (user explicitly stated)
                result = {
                    "document_type": extracted.document_type or "",
                    "organizations": extracted.organizations or [],
                    "locations": extracted.locations or [],
                    "dates": extracted.dates or [],
                    "department": extracted.department or "",
                    "topics": extracted.topics or [],
                    "_confidence": {
                        "document_type": "high",
                        "organizations": "high",
                        "dates": "high",
                        "locations": "high",
                        "department": "high",
                    },
                }
                return result
            except Exception as exc:
                logger.warning(f"Groq metadata extraction failed: {exc}, disabling LLM extractor.")
                self._groq_extractor = None

        if self._hf_extractor:
            try:
                extracted = self._hf_extractor.extract(query)
                result = {
                    "document_type": extracted.document_type,
                    "organizations": extracted.organizations,
                    "locations": extracted.locations,
                    "dates": extracted.dates,
                    "department": extracted.department,
                    "topics": extracted.topics,
                    "_confidence": {
                        "document_type": "high",
                        "organizations": "high",
                        "dates": "high",
                        "locations": "high",
                        "department": "high",
                    },
                }
                return result
            except Exception as exc:
                logger.warning(f"HF metadata extraction failed: {exc}, using rule-based.")

        return self._extract_rule_based(query)

    def _extract_rule_based(self, query: str) -> Dict[str, Any]:
        """
        Conservative rule-based extraction.

        Each extracted field is annotated with a confidence level:
          "high"   — the user explicitly requested a document-scoped filter
          "medium" — likely a filter but not certain
          "low"    — heuristic mention; should NOT be applied as a hard filter

        The QueryPlanner is responsible for honouring only HIGH-confidence
        fields as hard Qdrant/BM25 filters.
        """
        q = query.lower()
        confidence: Dict[str, str] = {}
        extracted: Dict[str, Any] = {
            "document_type": "",
            "organizations": [],
            "locations": [],
            "dates": [],
            "department": "",
            "topics": [],
        }

        # ── Document type ──────────────────────────────────────────────────
        # Explicit document-type requests ("find resumes", "show papers", …)
        if "resume" in q or "cv" in q:
            extracted["document_type"] = "resume"
            confidence["document_type"] = "high"
        elif "application form" in q or "application instruction" in q:
            extracted["document_type"] = "application instructions"
            confidence["document_type"] = "high"
        elif "research paper" in q or "nips" in q:
            extracted["document_type"] = "research_paper"
            confidence["document_type"] = "high"
        elif ("paper" in q or "publication" in q) and any(
            kw in q for kw in ("show", "find", "retrieve", "published", "authored")
        ):
            extracted["document_type"] = "research_paper"
            confidence["document_type"] = "medium"
        elif "paper" in q:
            # "paper" alone inside a content question is NOT a type filter
            # e.g. "What did the paper say about attention?" — keep as topic only
            confidence["document_type"] = "low"
        elif ("report" in q or "financial report" in q) and any(
            kw in q for kw in ("show", "find", "retrieve")
        ):
            extracted["document_type"] = "report"
            confidence["document_type"] = "medium"

        # ── Years / dates ──────────────────────────────────────────────────
        years = re.findall(r"\b(19\d{2}|20\d{2})\b", query)
        if years:
            extracted["dates"] = list(set(years))
            # Apply as hard filter only when the user explicitly scopes by date
            if any(ph in q for ph in _DATE_FILTER_PHRASES):
                confidence["dates"] = "high"
            else:
                # Year appears as context (e.g. "the 2017 Transformer paper")
                # — treat as soft retrieval hint only
                confidence["dates"] = "low"

        # ── Organizations ──────────────────────────────────────────────────
        if "niti aayog" in q:
            extracted["organizations"].append("NITI Aayog")
            confidence["organizations"] = "high"
        elif "niti" in q and any(
            kw in q for kw in ("internship", "scheme", "guidelines", "policy", "aayog")
        ):
            extracted["organizations"].append("NITI Aayog")
            confidence["organizations"] = "medium"
        elif "niti" in q:
            extracted["organizations"].append("NITI Aayog")
            confidence["organizations"] = "low"

        if "google" in q:
            extracted["organizations"].append("Google")
            # Mentioning "Google researchers" is about paper authorship, not a doc filter
            existing = confidence.get("organizations", "low")
            # Don't upgrade confidence if already set by NITI
            if "organizations" not in confidence:
                confidence["organizations"] = "low"

        # ── Topics (soft signals only — never hard filters) ────────────────
        match = re.search(r"\b(?:about|regarding|on)\s+([a-zA-Z0-9\s\-]+)", query, re.IGNORECASE)
        if match:
            topic = match.group(1).strip()
            if topic:
                extracted["topics"].append(topic)

        extracted["_confidence"] = confidence
        return extracted
