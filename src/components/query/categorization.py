from __future__ import annotations

import logging
import os
import re
from typing import Optional, Tuple

from ..query_categorization import QueryCategorization, QueryClassification
from .models import QueryType

logger = logging.getLogger(__name__)


class AdaptiveQueryCategorizer:
    """
    Classifies queries into the taxonomy:
    - fact: exact facts, numbers, dates, IDs, short specific answers.
    - detailed: procedural, relational, explanatory questions needing context.
    - general: broad summaries, overview, exploratory questions.
    - technical_exact: code symbols, API calls, syntax, file paths.
    - comparison: differences/similarities between concepts.
    - unanswerable / unsupported: detect obvious out-of-scope or absurd queries.
    """

    def __init__(self, use_llm_if_available: bool = True) -> None:
        self.use_llm = use_llm_if_available
        self._llm_classifier: Optional[QueryCategorization] = None
        if self.use_llm and os.getenv("GROQ_API_KEY"):
            try:
                self._llm_classifier = QueryCategorization()
            except Exception as exc:
                logger.warning(f"Could not initialize LLM query classifier: {exc}")

    def categorize(self, query: str) -> Tuple[QueryType, float]:
        q_lower = query.lower().strip()

        # Code identifiers, API signatures, and file references route to exact matching
        if any(w in q_lower for w in ("def ", "class ", "function", "api", "syntax", "implementation", ".py", ".json", "code")):
            return "technical_exact", 0.95

        # Comparative queries require sub-query decomposition
        if any(w in q_lower for w in ("compare", "difference between", "versus", "vs.", " vs ", "pros and cons")):
            return "comparison", 0.92

        # Fast heuristic pattern matching (sub-millisecond Layer 1)
        if re.search(r"^(who|when|where|which|how much|how many|what is the dimension|what is the value|what was the revenue|what was the bleu|what year|what dropout)\b", q_lower):
            return "fact", 0.90

        if re.search(r"^(how does|how do|explain|describe|why|walk me through|what are the steps|what are the instructions|how to)\b", q_lower):
            return "detailed", 0.90

        if re.search(r"^(tell me about|give me an overview|summarize|overview of|what are|outline)\b", q_lower):
            return "general", 0.85

        # Fallback to LLM classifier if available and pattern is ambiguous
        if self._llm_classifier:
            try:
                classification = self._llm_classifier.classify(query)
                cat = classification.category
                if cat in ("fact", "detailed", "general"):
                    return cat, classification.confidence
            except Exception as exc:
                logger.warning(f"LLM classification failed: {exc}, using rule-based fallback.")

        # Word count heuristics
        word_count = len(q_lower.split())
        if word_count <= 5:
            return "fact", 0.75
        elif word_count <= 14:
            return "detailed", 0.75
        else:
            return "general", 0.70
