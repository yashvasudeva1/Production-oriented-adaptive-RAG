from __future__ import annotations

import re
from typing import List


def decompose_query(query: str) -> List[str]:
    """
    Decomposes multi-part and comparative queries into atomic sub-questions.
    Returns [query] if query is simple or atomic.
    """
    q = query.strip()
    if not q:
        return []

    # Check for comparison: "Compare X and Y" or "What is the difference between X and Y"
    comp_match = re.search(
        r"(?:compare|difference between|differences between)\s+([^,]+?)\s+(?:and|with|versus|vs\.?)\s+(.+)",
        q,
        re.IGNORECASE,
    )
    if comp_match:
        term_a = comp_match.group(1).strip(" ?.")
        term_b = comp_match.group(2).strip(" ?.")
        return [
            f"What is {term_a}?",
            f"What is {term_b}?",
            f"Comparison and differences between {term_a} and {term_b}",
        ]

    # Check for conjunction: "..., and how does ..., and when should ..."
    parts = re.split(r",\s*and\s+|\s+and\s+how\s+|\s+and\s+when\s+|\s+and\s+why\s+", q, flags=re.IGNORECASE)
    if len(parts) > 1 and all(len(p.strip()) > 10 for p in parts):
        sub_queries = []
        for p in parts:
            p_clean = p.strip(" ?.")
            if not re.match(r"^(what|how|why|when|where|who)\b", p_clean, re.I):
                p_clean = f"What is {p_clean}"
            sub_queries.append(f"{p_clean}?")
        return sub_queries

    return [q]
