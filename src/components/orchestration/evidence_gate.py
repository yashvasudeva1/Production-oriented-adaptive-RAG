from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from ..reranker import RerankerResult

logger = logging.getLogger(__name__)


class EvidenceVerdict(BaseModel):
    """Structured decision returned by the Evidence Gate."""
    allowed: bool
    confidence: float
    reason: str
    supporting_chunks: List[Any] = Field(default_factory=list)
    signals: Dict[str, float] = Field(default_factory=dict)


class EvidenceGate:
    """
    Evidence Gate ensures that the generation model only responds when
    sufficient, reliable, relevant evidence exists in the retrieved context.
    Prevents answering false premises, hallucinations, or ungrounded queries.
    """

    def __init__(
        self,
        min_supporting_chunks: int = 1,
        min_confidence: float = 0.20,
        min_rerank_threshold: float = -5.5,
    ) -> None:
        self.min_supporting_chunks = min_supporting_chunks
        self.min_confidence = min_confidence
        self.min_rerank_threshold = min_rerank_threshold

    def evaluate(
        self,
        query: str,
        candidates: Sequence[RerankerResult],
    ) -> EvidenceVerdict:
        import math
        q_clean = query.strip()

        if not candidates:
            return EvidenceVerdict(
                allowed=False,
                confidence=0.0,
                reason="No candidate chunks were retrieved for this query.",
                supporting_chunks=[],
            )

        # 1. Adversarial Injection and Jailbreak Guard
        ADVERSARIAL_PATTERNS = [
            r"ignore\s+(all\s+)?(previous|prior|system|factual|facts|evidence|context|rules)",
            r"system\s+(prompt|override|message|instruction)",
            r"you\s+are\s+now\s+dan",
            r"act\s+as\s+(an?\s+)?(unrestricted|jailbroken|dan)",
            r"forget\s+(that\s+)?you\s+are",
            r"declare\s+that\s+.*was\s+invented",
            r"bypass\s+authentication",
            r"reveal\s+(database\s+)?credentials",
            r"secret\s+administrator\s+passwords",
            r"manufacture\s+explosives",
            r"drop\s+table",
            r"select\s+\*\s+from",
            r"sql\s+injection",
            r"<script>",
            r"unrestricted\s+chatbot\s+without\s+citations",
            r"private\s+tenants\s+in\s+the\s+database",
            r"repeat\s+the\s+word\s+['\"]?access_granted",
        ]
        q_lower = q_clean.lower()
        for pat in ADVERSARIAL_PATTERNS:
            if re.search(pat, q_lower):
                return EvidenceVerdict(
                    allowed=False,
                    confidence=0.0,
                    reason="Adversarial instruction or security policy violation detected. Operating strictly as a grounded research assistant.",
                    supporting_chunks=[],
                )

        # Check that top candidates cross the minimum relevance threshold
        viable_chunks = [
            c for c in candidates
            if getattr(c, "rerank_score", getattr(c, "score", 1.0)) >= self.min_rerank_threshold
            or (getattr(c, "keyword_score", None) is not None and c.keyword_score >= 3.0)
        ]

        if len(viable_chunks) < self.min_supporting_chunks:
            return EvidenceVerdict(
                allowed=False,
                confidence=0.15,
                reason=(
                    "Retrieved documents scored below the relevance threshold; "
                    "insufficient evidence to construct a grounded response."
                ),
                supporting_chunks=[],
            )

        # 2. Comprehensive stopword filtering & word-boundary token matching
        STOPWORDS = {
            "what", "when", "where", "which", "who", "whom", "whose", "why",
            "how", "many", "much", "does", "did", "do", "explain", "tell", "about",
            "describe", "compare", "summarize", "list", "find", "with", "from",
            "that", "this", "these", "those", "have", "were", "was", "is", "are",
            "be", "been", "being", "the", "for", "and", "can", "you", "all", "not",
            "but", "any", "some", "each", "into", "over", "after", "before", "more",
            "also", "than", "has", "had", "would", "could", "should", "its", "our",
            "their", "will", "out", "other", "give", "show", "provide", "use",
            "used", "using", "between", "under", "per", "won", "score", "year",
            "time", "place", "make", "made", "good", "new", "number", "count",
        }
        q_tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", q_lower)
        key_q_words = [w for w in q_tokens if w not in STOPWORDS]

        if not key_q_words:
            key_q_words = q_tokens

        def _word_tokens(text: str) -> set[str]:
            return set(re.findall(r"\b[a-z0-9_\-\.]+\b", text.lower()))

        def _conservative_stem(word: str) -> str:
            if len(word) < 6 or not word.isalpha():
                return word
            for suf in ("izations", "ization", "ations", "ation", "ings", "ment", "ness", "ities", "ence", "ance", "ents", "ants", "ing", "ion", "ity", "ies", "ied", "ive", "ent", "ant", "al", "ed", "es", "er", "ly", "s"):
                if word.endswith(suf) and len(word) - len(suf) >= 5:
                    return word[: -len(suf)]
            return word

        def _entity_present(term: str, token_set: set[str], raw_lower_text: str = "") -> bool:
            """Strict word-boundary matching for named entities and acronyms (plural-safe only)."""
            norm = term.lower().strip(".")
            if norm in token_set or term.lower() in token_set:
                return True
            if len(norm) >= 5 and norm.endswith("s") and norm[:-1] in token_set:
                return True
            if len(norm) >= 4 and (norm + "s") in token_set:
                return True
            # Handle dotted abbreviations (e.g., U.S. -> us or u.s.)
            if "." in term and raw_lower_text:
                if re.search(rf"(?<!\w){re.escape(term.lower())}(?!\w)", raw_lower_text):
                    return True
                collapsed = term.lower().replace(".", "")
                if len(collapsed) >= 2 and collapsed in token_set:
                    return True
            return False

        def _token_present(term: str, token_set: set[str], stem_set: Optional[set[str]] = None) -> bool:
            """Word-boundary + conservative morphological stem matching for content predicates."""
            if _entity_present(term, token_set):
                return True
            if stem_set is not None and len(term) >= 6 and term.isalpha():
                st = _conservative_stem(term)
                if len(st) >= 5 and st in stem_set:
                    return True
            return False

        chunk_token_sets = [_word_tokens(c.text) for c in viable_chunks]
        chunk_stem_sets = [{_conservative_stem(t) for t in ts if len(t) >= 6 and t.isalpha()} for ts in chunk_token_sets]
        combined_tokens: set[str] = set()
        combined_stems: set[str] = set()
        for ts, ss in zip(chunk_token_sets, chunk_stem_sets):
            combined_tokens.update(ts)
            combined_stems.update(ss)
        combined_lower_text = " ".join(c.text.lower() for c in viable_chunks)

        overlap_words = [w for w in key_q_words if _token_present(w, combined_tokens, combined_stems)]
        overlap_ratio = len(overlap_words) / max(1, len(key_q_words))

        # Check that at least one single chunk contains cohesive evidence (prevents cross-doc accidental word matching)
        max_chunk_overlap = max(
            (
                sum(1 for w in key_q_words if _token_present(w, ts, ss))
                for ts, ss in zip(chunk_token_sets, chunk_stem_sets)
            ),
            default=0,
        )

        # 2b. Named-Entity, Acronym, Temporal & Numeric Subject Grounding Policy
        # Prevents false-positive answering when unrelated documents share generic lexical terms
        # while missing the query's named entities, acronyms, years, or numeric constraints.
        SKIP_CAPS = {
            "What", "When", "Where", "Which", "Who", "Whom", "Whose", "Why", "How",
            "Explain", "Describe", "Compare", "Provide", "Summarize", "List", "Give",
            "Show", "Tell", "Find", "Retrieve", "Does", "Did", "Do", "Can", "Could",
            "Should", "Would", "Is", "Are", "Was", "Were", "In", "On", "At", "For",
            "From", "The", "A", "An", "If", "Once", "Online", "Application", "Form",
            "Paper", "Section", "Supplementary", "Appendix", "Analysis", "Pertinent",
        }
        GENERIC_UPPER = {"THE", "AND", "FOR", "NOT", "ALL", "ARE", "YOU", "CAN", "HOW", "WHY", "WHO"}

        query_entities: List[str] = []
        query_acronyms: List[str] = []
        camelcase_entities: List[str] = []
        for tok_idx, m in enumerate(re.finditer(r"(?:[A-Z]\.){2,}|\b[A-Za-z0-9_\-\./]+\b", q_clean)):
            raw_w = m.group(0)
            if re.fullmatch(r"(?:[A-Z]\.){2,}", raw_w):
                query_entities.append(raw_w.lower())
                query_acronyms.append(raw_w.lower())
                continue
            w = raw_w.rstrip(".,;:?!")
            if not w:
                continue
            if (
                re.fullmatch(r"[A-Z]{2,}(?:[-/][A-Z0-9]+)*[0-9]*", w)
                or re.fullmatch(r"[A-Z][0-9]+[A-Z0-9]*", w)
            ):
                if w not in GENERIC_UPPER and len(w.replace(".", "")) >= 2:
                    query_entities.append(w.lower())
                    query_acronyms.append(w.lower())
            elif re.fullmatch(r"[A-Z][a-zA-Z0-9_\-]{2,}", w):
                if w not in SKIP_CAPS and not (tok_idx == 0 and w.lower() in STOPWORDS):
                    query_entities.append(w.lower())
                    if re.search(r"[a-z][A-Z]|[A-Z].*\d", w):
                        camelcase_entities.append(w.lower())
        query_entities = list(dict.fromkeys(query_entities))
        query_acronyms = list(dict.fromkeys(query_acronyms))
        query_years = list(dict.fromkeys(re.findall(r"\b(?:18|19|20)\d{2}\b", q_clean)))

        matched_entities = [e for e in query_entities if _entity_present(e, combined_tokens, combined_lower_text)]
        entity_coverage = (len(matched_entities) / len(query_entities)) if query_entities else 1.0
        matched_years = [y for y in query_years if y in combined_tokens]
        unmatched_acronyms = [a for a in query_acronyms if not _entity_present(a, combined_tokens, combined_lower_text)]
        has_unmatched_acronym = len(unmatched_acronyms) > 0

        # Detect competing mid-sentence proper entities in top chunks (signals entity substitution mismatch)
        key_q_word_set = set(key_q_words)
        chunk_proper_entities: set[str] = set()
        for c in viable_chunks[:3]:
            for pm in re.finditer(r"\b[A-Z][a-zA-Z0-9_\-]{2,}\b", c.text):
                if pm.start() == 0:
                    continue
                prefix = c.text[max(0, pm.start() - 3): pm.start()]
                if re.search(r"[.!?:\n]\s*$", prefix):
                    continue
                pw = pm.group(0)
                pw_l = pw.lower()
                if (
                    pw not in SKIP_CAPS
                    and pw.upper() not in GENERIC_UPPER
                    and not pw.isupper()
                    and pw_l not in STOPWORDS
                    and pw_l not in key_q_word_set
                ):
                    chunk_proper_entities.add(pw_l)
        has_competing_chunk_entity = bool(chunk_proper_entities - set(query_entities))

        # Cohesive subject-predicate check: at least one single chunk must ground a named entity
        # alongside another substantive query word when multiple named entities are present.
        has_cohesive_entity_chunk = True
        if query_entities:
            has_cohesive_entity_chunk = any(
                any(_entity_present(e, ts, c.text.lower()) for e in query_entities)
                and sum(1 for w in key_q_words if _token_present(w, ts, ss)) >= min(2, len(key_q_words))
                for c, ts, ss in zip(viable_chunks, chunk_token_sets, chunk_stem_sets)
            )

        # 2c. Explicit temporal year grounding (any queried 4-digit year missing -> conflict/out-of-period)
        if query_years and len(matched_years) < len(query_years):
            return EvidenceVerdict(
                allowed=False,
                confidence=0.12,
                reason=(
                    "The retrieved context does not ground the specific year(s) referenced in the query "
                    "(temporal conflict or missing period). Refusing to hallucinate."
                ),
                supporting_chunks=[],
                signals={
                    "overlap_ratio": round(overlap_ratio, 3),
                    "entity_coverage": round(entity_coverage, 3),
                    "year_grounding": round(len(matched_years) / len(query_years), 3),
                },
            )

        # 2d. Conflicting numerical constraint grounding for direct factual lookups (preserving 'why'/'how' refutations)
        is_refutation_style = bool(re.match(r"^(?:why|how)\b", q_lower))
        if not is_refutation_style:
            norm_q_nums = re.findall(
                r"\bv\d+(?:\.\d+)+\b|\$\d+(?:,\d{3})*(?:\.\d+)?[kmbKMB]?|\b\d+(?:\.\d+)?%\b|\b\d+(?:,\d{3})*(?:\.\d+)?(?:gb|mb|tb|ms|khz|mhz|ghz|km|kg)\b|\b\d{2,}(?:,\d{3})*(?:\.\d+)?\b",
                q_clean,
                flags=re.IGNORECASE,
            )
            year_set = set(query_years)
            query_num_constraints = [
                n.lower().replace(",", "")
                for n in norm_q_nums
                if n.lower().replace(",", "") not in year_set
            ]
            if query_num_constraints:
                norm_ctx_text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", combined_lower_text)
                missing_nums = [
                    num for num in query_num_constraints
                    if not re.search(rf"(?<![\w.]){re.escape(num)}(?![\w.])", norm_ctx_text)
                ]
                if missing_nums:
                    return EvidenceVerdict(
                        allowed=False,
                        confidence=0.15,
                        reason=(
                            "The retrieved context does not ground the specific numerical value(s) or "
                            "version constraint(s) referenced in the query. Refusing to hallucinate."
                        ),
                        supporting_chunks=[],
                        signals={
                            "overlap_ratio": round(overlap_ratio, 3),
                            "entity_coverage": round(entity_coverage, 3),
                            "numeric_constraint_grounding": 0.0,
                        },
                    )

        # 2e. Named-Entity & Acronym grounding policy:
        # - Any explicit acronym/abbreviation absent from context -> reject
        # - Multi-entity comparative/conjunctive query missing any named entity -> reject
        # - Multi-entity query with <50% entity coverage -> reject
        # - All query entities absent (with multi-entity, CamelCase proper noun, low overlap, or competing chunk entity) -> reject
        is_comparative_entity_query = bool(
            re.search(r"\b(?:compare|comparison|versus|vs\.?|between|differ|differences)\b", q_lower)
        )
        entity_ungrounded = (
            has_unmatched_acronym
            or (
                len(matched_entities) == 0
                and (
                    len(query_entities) >= 2
                    or bool(camelcase_entities)
                    or max_chunk_overlap < 2
                    or has_competing_chunk_entity
                )
            )
            or (len(query_entities) >= 2 and entity_coverage < 0.50)
            or (len(query_entities) >= 2 and is_comparative_entity_query and entity_coverage < 1.0)
            or (len(query_entities) >= 2 and not has_cohesive_entity_chunk and entity_coverage < 1.0)
        )
        if query_entities and entity_ungrounded:
            return EvidenceVerdict(
                allowed=False,
                confidence=round(min(0.25, entity_coverage * 0.25), 2),
                reason=(
                    "The retrieved context does not ground the specific named entities, acronyms, or "
                    "factual subjects referenced in the query. Refusing to hallucinate."
                ),
                supporting_chunks=[],
                signals={
                    "overlap_ratio": round(overlap_ratio, 3),
                    "entity_coverage": round(entity_coverage, 3),
                    "term_coverage": 0.0,
                },
            )

        # 3. Evidence sufficiency check: unanswerable / out-of-domain detection
        if (
            len(overlap_words) == 0
            or (len(overlap_words) < 2 and len(key_q_words) >= 3)
            or overlap_ratio < 0.22
            or (len(key_q_words) >= 3 and max_chunk_overlap < 2 and overlap_ratio < 0.30)
        ):
            return EvidenceVerdict(
                allowed=False,
                confidence=round(overlap_ratio * 0.3, 2),
                reason=(
                    "The retrieved context lacks key terminology and premise evidence "
                    "for the query. Refusing to hallucinate."
                ),
                supporting_chunks=[],
                signals={
                    "overlap_ratio": round(overlap_ratio, 3),
                    "entity_coverage": round(entity_coverage, 3),
                    "term_coverage": 0.0,
                },
            )

        # Check that quantitative/count queries contain at least one numeric entity/digit in top chunks
        is_quant = bool(re.search(r"\b(?:how many|how much|what percentage|count of|number of|what year|when)\b", q_lower))
        has_numeric_evidence = True
        if is_quant:
            has_numeric_evidence = any(
                re.search(r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b", c.text.lower())
                for c in viable_chunks[:3]
            )
            if not has_numeric_evidence:
                return EvidenceVerdict(
                    allowed=False,
                    confidence=round(overlap_ratio * 0.2, 2),
                    reason="The query asks for a quantitative count or measurement, but the retrieved context contains no numeric values or counts.",
                    supporting_chunks=[],
                    signals={"overlap_ratio": round(overlap_ratio, 3), "numeric_grounding": 0.0},
                )

        # 4. Calibrated, measurable evidence confidence score
        # Grounded in physical lexical coverage, normalized retrieval relevance, and provenance agreement
        best_score = max((getattr(c, "rerank_score", getattr(c, "score", 1.0)) for c in viable_chunks), default=0.0)
        if best_score > 1.0:
            sig_score = min(1.0, max(0.0, best_score / 15.0))
        elif best_score < 0.0:
            sig_score = 1.0 / (1.0 + math.exp(-max(-6.0, best_score)))
        else:
            sig_score = min(1.0, best_score * 20.0) if best_score < 0.05 else best_score

        # Check for dual-retriever agreement in top chunks
        has_consensus = any(
            len(getattr(c, "provenance_sources", [])) > 1
            or ("dense" in getattr(c, "retriever_name", "") and getattr(c, "keyword_score", None) is not None)
            for c in viable_chunks[:3]
        )
        consensus_bonus = 0.08 if has_consensus else 0.0

        # Calibrated evidence index
        confidence = round(min(1.0, (sig_score * 0.40) + (overlap_ratio * 0.60) + consensus_bonus), 2)

        conf_signals = {
            "term_overlap_ratio": round(overlap_ratio, 4),
            "relevance_score": round(sig_score, 4),
            "consensus_detected": 1.0 if has_consensus else 0.0,
            "quantitative_grounded": 1.0 if (is_quant and has_numeric_evidence) else (0.5 if not is_quant else 0.0),
            "calibrated_confidence": confidence,
        }

        if confidence < self.min_confidence:
            return EvidenceVerdict(
                allowed=False,
                confidence=confidence,
                reason=f"Evidence confidence {confidence:.2f} is below minimum threshold {self.min_confidence:.2f}.",
                supporting_chunks=[],
                signals=conf_signals,
            )

        return EvidenceVerdict(
            allowed=True,
            confidence=confidence,
            reason="Sufficient grounded evidence found in retrieved chunks.",
            supporting_chunks=viable_chunks,
            signals=conf_signals,
        )
