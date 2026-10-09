from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Literal, Set

QueryIntent = Literal[
    "fact",
    "procedural",
    "detailed",
    "comparison",
    "summary",
    "technical_exact",
    "general",
]

InitialRoute = Literal["fast", "balanced", "deep"]


@dataclass
class QuerySignals:
    """
    Sub-millisecond deterministic signals extracted without an LLM.
    Provides structural guidance for the confidence-based retrieval cascade.
    """
    raw_query: str
    normalized_query: str
    intent: QueryIntent
    initial_route: InitialRoute
    complexity: Literal["low", "medium", "high"]

    exact_identifiers: List[str] = field(default_factory=list)
    quoted_phrases: List[str] = field(default_factory=list)
    code_symbols: List[str] = field(default_factory=list)
    dates_and_years: List[str] = field(default_factory=list)
    section_references: List[str] = field(default_factory=list)
    comparison_terms: List[str] = field(default_factory=list)
    is_multi_part: bool = False
    is_multi_hop: bool = False
    word_count: int = 0
    clause_count: int = 1


# Common academic and technical parameter identifiers
KNOWN_TECHNICAL_IDENTIFIERS: Set[str] = {
    "d_model",
    "d_k",
    "d_v",
    "d_ff",
    "p_drop",
    "bleu",
    "adam",
    "transformer",
    "self-attention",
    "multi-head",
    "layer_norm",
    "layernorm",
    "p100",
    "bpe",
    "niti",
    "aayog",
    "nips",
}


def extract_query_signals(query: str, normalized_query: str) -> QuerySignals:
    """
    Extract deterministic lexical, structural, and syntactic signals from query text.
    Executes in under 0.2 milliseconds with zero network or model overhead.
    """
    q_raw = query.strip()
    q_norm = normalized_query.strip()
    q_lower = q_norm.lower()

    words = q_lower.split()
    word_count = len(words)

    # 1. Quoted literal phrases
    quoted = re.findall(r'["\']([^"\']+)["\']', q_raw)

    # 2. Exact identifiers & parameters
    exact_ids: List[str] = []
    for token in re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", q_lower):
        if token in KNOWN_TECHNICAL_IDENTIFIERS:
            exact_ids.append(token)
    # Numbers and percentages (e.g. 512, 64, 85%, 0.1)
    numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", q_lower)
    exact_ids.extend(numbers)
    exact_ids = list(dict.fromkeys(exact_ids))

    # 3. Code symbols & syntax keywords
    code_symbols: List[str] = []
    for sym in ("def ", "class ", ".py", ".json", ".xml", "api", "function", "method", "syntax", "endpoint"):
        if sym in q_lower:
            code_symbols.append(sym)

    # 4. Dates and years (e.g. 2017, 2024)
    dates = re.findall(r"\b(19\d{2}|20\d{2})\b", q_norm)
    dates = list(set(dates))

    # 5. Section, table, page references
    section_refs: List[str] = []
    for sec_word in ("section", "table", "figure", "page", "appendix"):
        if re.search(rf"\b{sec_word}\s+[0-9a-zA-Z]+", q_lower):
            section_refs.append(sec_word)

    # 6. Comparison indicators
    comp_matches = re.findall(
        r"\b(?:compare|difference between|differences between|versus|vs\.?|pros and cons)\b",
        q_lower,
    )

    # 7. Multi-part conjunction indicators (distinct independent question clauses)
    conjunction_patterns = [
        r",\s*and\s+(?:what|how|why|when|where|who)\b",
        r"\?\s+[A-Z]",
    ]
    is_multi_part = any(re.search(p, q_raw) for p in conjunction_patterns) or len(re.findall(r"\?", q_raw)) > 1

    # Multi-hop synthesis indicators (connecting disparate concepts)
    multi_hop_patterns = [
        r"\b(?:connect to|connects to|interact with|interact in|jointly impact|jointly affect|influence whether|influence how|mitigate\b.*\bwhen|prevent\b.*\bcaused by|combine\b.*\bfrom)\b",
    ]
    is_multi_hop = any(re.search(p, q_lower) for p in multi_hop_patterns)

    # Clause count estimate
    clauses = re.split(r"[,;]|\band\b", q_lower)
    clause_count = max(1, len([c for c in clauses if len(c.strip()) > 3]))

    # 8. Intent classification via deterministic precedence
    if comp_matches:
        intent: QueryIntent = "comparison"
    elif is_multi_part:
        intent = "detailed"
    elif code_symbols:
        intent = "technical_exact"
    elif re.search(r"\b(?:instructions to|steps to|procedure for|how to fill|how do i apply)\b", q_lower):
        intent = "procedural"
    elif re.search(r"^(?:who|when|where|which|how much|how many|what is the dimension|what is the value|what was the bleu|what year|what dropout)\b", q_lower):
        intent = "fact"
    elif re.search(r"\b(?:summarize|overview of|tell me about|core thesis|main contributions)\b", q_lower):
        intent = "summary"
    elif re.search(r"^(?:how does|how do|explain|describe|why|walk me through|what are the)\b", q_lower):
        intent = "detailed"
    elif exact_ids and word_count <= 8:
        intent = "fact"
    else:
        intent = "general"

    # 9. Complexity determination
    if comp_matches or is_multi_part or is_multi_hop or clause_count >= 3 or word_count > 20:
        complexity = "high"
    elif word_count > 10 or intent in ("detailed", "procedural", "summary"):
        complexity = "medium"
    else:
        complexity = "low"

    # 10. Initial route selection for the confidence cascade
    # Exact/simple queries begin on the ultra-fast BM25 path
    if (intent in ("fact", "technical_exact") and complexity == "low") or (quoted and word_count <= 6):
        initial_route: InitialRoute = "fast"
    elif complexity == "high" or intent in ("comparison", "summary") or is_multi_hop:
        initial_route = "deep"
    else:
        initial_route = "balanced"

    return QuerySignals(
        raw_query=q_raw,
        normalized_query=q_norm,
        intent=intent,
        initial_route=initial_route,
        complexity=complexity,
        exact_identifiers=exact_ids,
        quoted_phrases=quoted,
        code_symbols=code_symbols,
        dates_and_years=dates,
        section_references=section_refs,
        comparison_terms=comp_matches,
        is_multi_part=is_multi_part,
        is_multi_hop=is_multi_hop,
        word_count=word_count,
        clause_count=clause_count,
    )
