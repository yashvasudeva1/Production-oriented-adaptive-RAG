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
    is_quantitative: bool = False
    word_count: int = 0
    clause_count: int = 1


# Common academic and technical parameter identifiers
KNOWN_TECHNICAL_IDENTIFIERS: Set[str] = {
    "d_model",
    "d_k",
    "d_v",
    "d_ff",
    "p_drop",
    "d_m",
    "p_dropout",
    "layernorm",
    "layer_norm",
    "bleu",
    "bpe",
    "adam",
    "p100",
    "v100",
    "hnsw",
    "bm25",
    "rrf",
}

DISALLOWED_CAPS: Set[str] = {
    "A", "I", "IN", "ON", "AT", "TO", "FOR", "OF", "BY", "IS", "ARE", "WAS", "WERE",
    "WHAT", "WHEN", "WHERE", "WHY", "HOW", "WHICH", "WHO", "THE", "AND", "BUT", "OR",
    "NOT", "ALL", "ANY", "CAN", "DO", "DID", "HAS", "HAVE", "HAD", "BE", "BEEN", "WITH"
}

ENGLISH_STOP_WORDS: Set[str] = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "can", "could", "should", "would", "may", "might", "must",
    "what", "which", "who", "whom", "this", "that", "these", "those", "where",
    "when", "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "and", "but", "if", "or", "because", "as",
    "until", "while", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "up", "down", "out", "off", "over", "under",
    "again", "further", "then", "once", "here", "there"
}


KNOWN_TECHNICAL_ACRONYMS: Set[str] = {
    "BLEU", "HNSW", "RRF", "BM25", "BERT", "LSTM", "GRU", "GPU", "TPU", "CPU",
    "BPE", "CNN", "RNN", "SGD", "NLP", "FFN", "MHA", "ROUGE", "GLUE"
}


def extract_technical_identifiers(query: str) -> List[str]:
    """
    Extract exact technical identifiers, formulas, acronyms, camelCase/snake_case
    symbols, alphanumeric hardware/model identifiers, and single-character mathematical
    variables from the query string without classifying ordinary English prose.
    """
    found: List[str] = []

    # 1. Snake_case tokens (e.g. d_model, p_drop, layer_norm, attn_weights)
    snake_case = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*_[a-zA-Z0-9_]+\b", query)
    found.extend(snake_case)

    # 2. CamelCase / mixed-case tokens (e.g. ByteNet, ConvS2S, Word2Vec, RoBERTa)
    camel_case = re.findall(r"\b[A-Za-z0-9]*[a-z][A-Z][a-zA-Z0-9]*\b", query)
    found.extend(camel_case)

    # 3. Alphanumeric model/hardware names (e.g. P100, V100, GPT4, Llama3, bge-small)
    alphanum = re.findall(r"\b(?:[A-Za-z]+[0-9]+|[0-9]+[A-Za-z]+)[A-Za-z0-9_-]*\b", query)
    found.extend(alphanum)

    # 4. Technical uppercase acronyms (e.g. BLEU, HNSW, RRF, BM25, BERT, LSTM, GPU, GPUs)
    raw_acronyms = re.findall(r"\b([A-Z]{2,6})s?\b", query)
    for acr in raw_acronyms:
        if acr in KNOWN_TECHNICAL_ACRONYMS:
            found.append(acr)

    # 5. Single-character mathematical variables (e.g. N in 'parameter N', 'heads h', 'where N = 6')
    # Standalone uppercase letters (excluding 'A' and 'I')
    single_upper = re.findall(r"\b([B-HJ-Z])\b", query)
    found.extend(single_upper)

    # Lowercase single-letter mathematical variables when accompanied by variable/parameter context
    single_var_matches = re.findall(
        r"(?:heads?|layers?|dimension|parameter|variable|value\s+of|where)\s+([a-zA-Z])\b",
        query,
        re.IGNORECASE,
    )
    for v in single_var_matches:
        if v.lower() not in {"a", "i"}:
            found.append(v)

    # Explicit variable assignments like 'h = 8' or 'k = 5' or 'N = 6'
    assignments = re.findall(r"\b([a-zA-Z])\s*=\s*\d+", query)
    for v in assignments:
        if v.lower() not in {"a", "i"}:
            found.append(v)

    # 6. Dotted code identifiers (e.g. indexer.py, data.json)
    dotted = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\.(?:py|json|xml|csv|tsv|md)\b", query)
    found.extend(dotted)

    # 7. Check curated symbols (case-insensitive)
    for sym in KNOWN_TECHNICAL_IDENTIFIERS:
        if re.search(r"\b" + re.escape(sym) + r"\b", query, re.IGNORECASE):
            found.append(sym)

    # Deduplicate preserving case
    dedup: List[str] = []
    seen = set()
    for item in found:
        norm = item.strip()
        if norm and norm.lower() not in seen and norm.lower() not in ENGLISH_STOP_WORDS:
            seen.add(norm.lower())
            dedup.append(norm)

    return dedup


def extract_query_signals(query: str, normalized_query: Optional[str] = None) -> QuerySignals:
    """
    Extract deterministic lexical, structural, and syntactic signals from query text.
    Executes in under 0.2 milliseconds with zero network or model overhead.
    """
    q_raw = query.strip()
    q_norm = (normalized_query if normalized_query is not None else query).strip()
    q_lower = q_norm.lower()

    words = q_lower.split()
    word_count = len(words)

    # 1. Quoted literal phrases (distinguish true quotes from English possessives/contractions)
    d_quotes = re.findall(r'"([^"\r\n]+)"', q_raw)
    s_quotes = re.findall(r"(?:^|[\s(])'([^'\r\n]+)'(?:$|[\s).,!?;:])", q_raw)
    quoted = [
        item.strip()
        for item in (d_quotes + s_quotes)
        if len(item.strip()) > 1 and len(item.split()) <= 12
    ]

    # 2. Exact identifiers & parameters (audited detector: symbols, formulas, acronyms, single-char vars)
    exact_ids = extract_technical_identifiers(q_raw)

    # Numbers and percentages for quantitative queries
    numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", q_lower)

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

    # Quantitative count/measurement indicators
    is_quantitative = bool(
        re.search(
            r"\b(?:how many|how much|how often|what percentage|count of|number of|what year|when)\b",
            q_lower,
        )
    )

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
    # Quantitative fact questions without explicit identifiers require dense semantic alignment
    if is_quantitative and not exact_ids and not quoted:
        initial_route: InitialRoute = "balanced"
    elif (intent in ("fact", "technical_exact") and complexity == "low") or (quoted and word_count <= 6):
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
        is_quantitative=is_quantitative,
        word_count=word_count,
        clause_count=clause_count,
    )
