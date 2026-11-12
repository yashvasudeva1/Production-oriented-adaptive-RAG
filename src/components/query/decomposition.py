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

    # Check for conjunction: "..., and how does ...", "and her ...", "and his ...", "and what ..."
    parts = re.split(
        r",\s*and\s+|\s+and\s+(?:how|when|why|where|what|who|which|her|his|their|its|the)\s+",
        q,
        flags=re.IGNORECASE,
    )
    if len(parts) > 1 and all(len(p.strip()) > 8 for p in parts):
        sub_queries = []
        for p in parts:
            p_clean = p.strip(" ?.")
            # Remove hypothetical conditionals like "if my future wife has ..."
            p_clean = re.sub(r"^(?:if\s+[\w\s]+?\s+has\s+)", "", p_clean, flags=re.I).strip()
            if not re.match(r"^(what|how|why|when|where|who|which)\b", p_clean, re.I):
                p_clean = f"What is {p_clean}"
            sub_queries.append(f"{p_clean}?")
        return sub_queries

    # Check for multi-hop synthesis: "How does A connect to / interact with / influence B"
    hop_match = re.search(
        r"(?:how does|how do|explain how|why does)\s+(.+?)\s+(?:connect to|connects to|interact with|interact in|jointly impact|jointly affect|influence whether|influence|prevent|mitigate)\s+(.+)",
        q,
        re.IGNORECASE,
    )
    if hop_match:
        part_a = hop_match.group(1).strip(" ?.")
        part_b = hop_match.group(2).strip(" ?.")
        sub_queries = []
        if not re.match(r"^(what|how|why|when|where|who)\b", part_a, re.I):
            sub_queries.append(f"What is {part_a}?")
        else:
            sub_queries.append(f"{part_a}?")
        if not re.match(r"^(what|how|why|when|where|who)\b", part_b, re.I):
            sub_queries.append(f"What is {part_b}?")
        else:
            sub_queries.append(f"{part_b}?")
        sub_queries.append(q)
        return sub_queries

    return [q]
