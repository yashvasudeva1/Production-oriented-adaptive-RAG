from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

from ..query_metadata_extraction import QueryMetadata, QueryMetadataExtractor

logger = logging.getLogger(__name__)


class AdaptiveQueryMetadataExtractor:
    """
    Extracts explicit metadata constraints from user queries.
    Uses open source models via Groq or Hugging Face, with
    deterministic regex-based extraction as a fallback.
    """

    def __init__(self, use_llm_if_available: bool = True) -> None:
        self.use_llm = use_llm_if_available
        self._groq_extractor: Optional[Any] = None
        self._hf_extractor: Optional[QueryMetadataExtractor] = None

        if self.use_llm and os.getenv("GROQ_API_KEY"):
            try:
                from langchain_groq import ChatGroq
                model_name = os.getenv("GROQ_MODEL", os.getenv("LLM_MODEL", "qwen/qwen3.8-27b"))
                self._groq_extractor = ChatGroq(
                    model=model_name,
                    api_key=os.getenv("GROQ_API_KEY"),
                    temperature=0.0,
                    max_retries=2,
                ).with_structured_output(QueryMetadata)
            except Exception as exc:
                logger.warning(f"Could not initialize Groq metadata extractor: {exc}")

        if self.use_llm and os.getenv("HF_TOKEN") and not self._groq_extractor:
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
                return {
                    "document_type": extracted.document_type or "",
                    "organizations": extracted.organizations or [],
                    "locations": extracted.locations or [],
                    "dates": extracted.dates or [],
                    "department": extracted.department or "",
                    "topics": extracted.topics or [],
                }
            except Exception as exc:
                logger.warning(f"Groq metadata extraction failed: {exc}, trying fallback.")
                if "429" in str(exc) or "rate_limit" in str(exc).lower():
                    self._groq_extractor = None

        if self._hf_extractor:
            try:
                extracted = self._hf_extractor.extract(query)
                return {
                    "document_type": extracted.document_type,
                    "organizations": extracted.organizations,
                    "locations": extracted.locations,
                    "dates": extracted.dates,
                    "department": extracted.department,
                    "topics": extracted.topics,
                }
            except Exception as exc:
                logger.warning(f"HF metadata extraction failed: {exc}, using rule-based.")

        return self._extract_rule_based(query)

    def _extract_rule_based(self, query: str) -> Dict[str, Any]:
        q = query.lower()
        extracted: Dict[str, Any] = {
            "document_type": "",
            "organizations": [],
            "locations": [],
            "dates": [],
            "department": "",
            "topics": [],
        }

        # Document type hints
        if "research paper" in q or "paper" in q or "nips" in q:
            extracted["document_type"] = "research_paper"
        elif "resume" in q or "cv" in q:
            extracted["document_type"] = "resume"
        elif "report" in q or "financial report" in q:
            extracted["document_type"] = "report"
        elif "instruction" in q or "application form" in q:
            extracted["document_type"] = "application instructions"

        # Years and dates
        years = re.findall(r"\b(19\d{2}|20\d{2})\b", query)
        if years:
            extracted["dates"] = list(set(years))

        # Known organizations in dataset
        if "niti aayog" in q or "niti" in q:
            extracted["organizations"].append("NITI Aayog")
        if "google" in q:
            extracted["organizations"].append("Google")

        # Topics extraction (terms after 'about', 'regarding', 'for')
        match = re.search(r"\b(?:about|regarding|on)\s+([a-zA-Z0-9\s\-]+)", query, re.IGNORECASE)
        if match:
            topic = match.group(1).strip()
            if topic:
                extracted["topics"].append(topic)

        return extracted
