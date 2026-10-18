from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence
from pydantic import BaseModel, Field

from ..reranker import RerankerResult


class CitationSource(BaseModel):
    """Traceable citation source metadata."""
    citation_id: int
    document_id: str
    filename: str
    chunk_id: str
    page: Optional[int] = None
    section: Optional[str] = None
    source_locator: Optional[str] = None
    snippet: str = ""
    text: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ClaimVerification(BaseModel):
    """Claim-level verification mapping an atomic statement to its supporting evidence."""
    claim_id: int
    statement: str
    cited_source_id: Optional[int] = None
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    status: str = "supported"  # "supported", "partially_supported", "ungrounded"
    support_score: float = 1.0
    evidence_span: str = ""


class CitationValidationReport(BaseModel):
    """Results of citation validation and independent claim-level evidence tracking."""
    is_valid: bool
    cited_ids: List[int] = Field(default_factory=list)
    valid_citations: List[int] = Field(default_factory=list)
    invalid_citations: List[int] = Field(default_factory=list)
    hallucinated_citations: List[int] = Field(default_factory=list)
    claim_verifications: List[ClaimVerification] = Field(default_factory=list)
    supported_claims_ratio: float = 1.0
    factual_support_ratio: float = 1.0
    is_factually_supported: bool = True
    has_citations: bool = False
    citation_precision: float = 1.0
    citation_alignment_score: float = 1.0
    unsupported_claims: List[str] = Field(default_factory=list)
    details: str = ""


class CitationBuilder:
    """Builds numbered citation references [1], [2] from supporting chunks."""

    def build_sources(
        self, chunks: Sequence[Any]
    ) -> List[CitationSource]:
        sources: List[CitationSource] = []
        for idx, chunk in enumerate(chunks, start=1):
            meta = getattr(chunk, "metadata", None) or (chunk.get("metadata", {}) if isinstance(chunk, dict) else {}) or {}
            did = getattr(chunk, "document_id", "") or (chunk.get("document_id", "") if isinstance(chunk, dict) else "")
            cid = getattr(chunk, "chunk_id", "") or (chunk.get("chunk_id", "") if isinstance(chunk, dict) else "")
            text = getattr(chunk, "text", "") or (chunk.get("text", "") if isinstance(chunk, dict) else "")
            page = getattr(chunk, "page", None) or (chunk.get("page", None) if isinstance(chunk, dict) else None)
            section = getattr(chunk, "section", None) or (chunk.get("section", None) if isinstance(chunk, dict) else None)
            locator = getattr(chunk, "source_locator", None) or (chunk.get("source_locator", None) if isinstance(chunk, dict) else None)
            fn = meta.get("filename") or (meta.get("metadata", {}).get("filename") if isinstance(meta.get("metadata"), dict) else None)
            if not fn and did:
                try:
                    from ..registry import SystemRegistry
                    doc = SystemRegistry.get_doc_store().get(str(did))
                    if doc and doc.filename:
                        fn = doc.filename
                except Exception:
                    pass
            if not fn:
                fn = f"doc_{str(did)[:8]}"
            snippet = text[:600].replace("\n", " ").strip()
            sources.append(
                CitationSource(
                    citation_id=idx,
                    document_id=str(did),
                    filename=fn,
                    chunk_id=str(cid),
                    page=page,
                    section=section,
                    source_locator=locator,
                    snippet=snippet,
                    text=text,
                    metadata=meta,
                )
            )
        return sources

    def format_sources_for_prompt(self, sources: Sequence[CitationSource], chunks: Sequence[Any]) -> str:
        blocks: List[str] = []
        for src, chunk in zip(sources, chunks):
            locator = src.source_locator or f"doc:{src.filename}"
            txt = getattr(chunk, "text", "") or (chunk.get("text", "") if isinstance(chunk, dict) else "")
            blocks.append(
                f"Source [{src.citation_id}] ({src.filename}, {locator}):\n{txt}"
            )
        return "\n\n".join(blocks)


class CitationValidator:
    """
    Validates that every citation claim in generated text references an existing,
    retrieved chunk, and that the claimed information is physically present.
    """

    CITATION_PATTERN = re.compile(r"\[(\d+)\]")
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "as", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "can", "could",
        "will", "would", "shall", "should", "may", "might", "must", "it",
        "its", "they", "them", "their", "this", "that", "these", "those",
        "which", "who", "whom", "whose", "what", "where", "when", "why",
        "how", "rely", "relies", "use", "uses", "used", "also"
    }

    NEGATION_WORDS = {
        "not", "no", "never", "none", "neither", "nor", "cannot", "cant",
        "without", "hardly", "barely", "scarcely", "refuse", "refuses",
        "fails", "failed", "doesnt", "isnt", "arent", "wasnt", "werent",
        "havent", "hasnt", "wont", "wouldnt", "dont", "didnt"
    }

    CONTRADICTION_PAIRS = [
        ({"unpaid", "honorary", "free", "non-salaried"}, {"paid", "salary", "salaried", "stipend", "remuneration"}),
        ({"dispensed", "without", "eliminated", "removed", "no"}, {"uses", "using", "required", "employs", "relies", "incorporates"}),
        ({"maximum", "max"}, {"minimum", "min"}),
        ({"faster", "speedup", "quick"}, {"slower", "slow", "bottleneck"}),
        ({"prohibits", "disallows", "forbidden", "ineligible"}, {"allows", "permits", "eligible", "accepted"}),
        ({"parallel", "parallelizable"}, {"sequential", "recurrent", "autoregressive"}),
    ]

    ENTITY_EXCLUSIONS = {
        "The", "This", "That", "These", "Those", "There", "Here", "It", "Its",
        "We", "They", "Our", "Their", "According", "Based", "In", "On", "At",
        "For", "With", "By", "As", "If", "When", "While", "Sources", "Source",
        "Figure", "Table", "Section", "Chapter", "Paper", "Papers", "Model",
        "Models", "Transformer", "Transformers", "Attention", "System", "Systems",
        "Method", "Methods", "Data", "Document", "Documents"
    }

    def _extract_content_words(self, text: str) -> List[str]:
        words = re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", text.lower())
        return [w for w in words if w not in self.STOPWORDS]

    def _check_numeric_contradiction(
        self, claim_text: str, ctx_texts: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """Verify that numbers/quantities asserted in the claim exist in the evidence context."""
        claim_nums = re.findall(r"\b\d+(?:\.\d+)?%?\b", claim_text)
        if not claim_nums:
            return False, None

        all_ctx_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", " ".join(ctx_texts)))
        if not all_ctx_nums:
            return True, f"Claim asserts numbers {claim_nums} but evidence context contains no numeric data"

        norm_ctx_nums = {n.replace(",", "").replace("%", "") for n in all_ctx_nums}
        missing_nums = [
            n for n in claim_nums
            if n not in all_ctx_nums and n.replace(",", "").replace("%", "") not in norm_ctx_nums
        ]

        if missing_nums:
            return True, f"Numeric contradiction: {missing_nums} not present in evidence context"
        return False, None

    def _check_polarity_inversion(
        self, claim_text: str, evidence_span: str, all_ctx_text: str
    ) -> Tuple[bool, Optional[str]]:
        """Detect negation or antonym contradictions reversing factual polarity."""
        c_clean = re.sub(r"[^\w\s]", "", claim_text.lower())
        c_words = set(c_clean.split())
        ctx_words = set(re.sub(r"[^\w\s]", "", all_ctx_text.lower()).split())

        # 1. Antonym contradiction check
        for set_a, set_b in self.CONTRADICTION_PAIRS:
            if (c_words & set_a and ctx_words & set_b) or (c_words & set_b and ctx_words & set_a):
                return True, "Antonym contradiction between claim and evidence"

        # 2. Negation polarity inversion against matching evidence span
        if not evidence_span:
            return False, None

        e_clean = re.sub(r"[^\w\s]", "", evidence_span.lower())
        e_words = set(e_clean.split())

        c_has_neg = bool(c_words & self.NEGATION_WORDS)
        e_has_neg = bool(e_words & self.NEGATION_WORDS)

        if c_has_neg != e_has_neg:
            c_content = self._extract_content_words(claim_text)
            e_content = self._extract_content_words(evidence_span)
            if c_content and e_content:
                overlap = len(set(c_content) & set(e_content)) / len(set(c_content))
                if overlap >= 0.40:
                    return True, "Polarity inversion: negation mismatch on evidence statement"

        return False, None

    def _check_entity_swapping(
        self, claim_text: str, all_ctx_text: str
    ) -> Tuple[bool, Optional[str]]:
        """Detect ungrounded or swapped named entities not present in context."""
        words = claim_text.strip().split()
        first_word = re.sub(r"[^\w]", "", words[0]) if words else ""

        capitalized = re.findall(r"\b[A-Z][a-zA-Z0-9_\-]+\b", claim_text)
        candidate_entities = [
            e for e in capitalized
            if e not in self.ENTITY_EXCLUSIONS
            and len(e) > 2
            and not (e == first_word and e.istitle() and not any(c.isupper() for c in e[1:]))
        ]
        if not candidate_entities:
            return False, None

        ctx_lower = all_ctx_text.lower()
        missing = [
            e for e in candidate_entities
            if e.lower() not in ctx_lower and (len(e) <= 3 or e.lower().rstrip("s") not in ctx_lower)
        ]

        if missing:
            return True, f"Entity mismatch: {missing} not grounded in evidence context"
        return False, None

    def _check_unsupported_predicates(
        self, claim_text: str, all_ctx_text: str
    ) -> Tuple[bool, Optional[str]]:
        """Detect claims that borrow vocabulary from evidence but introduce ungrounded predicates."""
        c_content = self._extract_content_words(claim_text)
        if not c_content or len(c_content) < 3:
            return False, None

        ctx_lower = all_ctx_text.lower()
        absent_content = [w for w in c_content if w not in ctx_lower]

        if len(absent_content) / len(c_content) >= 0.50:
            return True, f"Unsupported predicate: key terms {absent_content} ungrounded in context"
        return False, None

    def validate_claim_support(
        self,
        answer_text: str,
        context_chunks: Sequence[Any],
    ) -> Dict[str, Any]:
        """
        Validates claim support independently of citation existence.

        Extracts atomic sentences from answer_text and evaluates whether the factual
        content of each claim is grounded in the retrieved context chunks, regardless
        of whether the text contains bracketed citation tokens.
        """
        if not answer_text or not answer_text.strip():
            return {
                "claims_count": 0,
                "supported_claims_count": 0,
                "factual_support_ratio": 1.0,
                "is_supported": True,
                "claim_details": [],
                "unsupported_claims": [],
            }

        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", answer_text)
            if len(s.strip()) > 8 and not s.strip().lower().startswith("sources:")
        ]
        if not sentences:
            return {
                "claims_count": 0,
                "supported_claims_count": 0,
                "factual_support_ratio": 1.0,
                "is_supported": True,
                "claim_details": [],
                "unsupported_claims": [],
            }

        ctx_texts: List[str] = []
        for c in context_chunks:
            if isinstance(c, str):
                ctx_texts.append(c)
            elif hasattr(c, "text") and c.text:
                ctx_texts.append(c.text)
            elif hasattr(c, "snippet") and c.snippet:
                ctx_texts.append(c.snippet)
            elif isinstance(c, dict):
                ctx_texts.append(c.get("text", "") or c.get("snippet", ""))

        all_ctx_text = " ".join(ctx_texts).lower()
        all_ctx_words = set(re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", all_ctx_text))

        claim_details: List[Dict[str, Any]] = []
        supported_count = 0
        unsupported: List[str] = []

        for idx, sent in enumerate(sentences, start=1):
            sent_clean = self.CITATION_PATTERN.sub("", sent).strip()
            content_words = self._extract_content_words(sent_clean)
            c_word_set = set(content_words)

            best_span_overlap = 0.0
            best_span = ""

            for ctx in ctx_texts:
                src_sents = re.split(r"(?<=[.!?])\s+", ctx)
                for s_sent in src_sents:
                    src_words = set(re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", s_sent.lower()))
                    if c_word_set and src_words:
                        overlap = len(c_word_set & src_words) / len(c_word_set)
                        if overlap > best_span_overlap:
                            best_span_overlap = overlap
                            best_span = s_sent.strip()

            # Global context overlap handles cross-sentence synthesis
            global_overlap = (len(c_word_set & all_ctx_words) / len(c_word_set)) if c_word_set else 1.0
            effective_overlap = max(best_span_overlap, global_overlap)

            # Adversarial and Integrity Checks
            is_num_conflict, num_err = self._check_numeric_contradiction(sent_clean, ctx_texts)
            is_neg_conflict, neg_err = self._check_polarity_inversion(sent_clean, best_span, all_ctx_text)
            is_entity_conflict, ent_err = self._check_entity_swapping(sent_clean, all_ctx_text)
            is_pred_conflict, pred_err = self._check_unsupported_predicates(sent_clean, all_ctx_text)

            failure_reason = ""
            if is_num_conflict:
                status = "contradicted"
                effective_overlap = 0.0
                failure_reason = num_err or "numeric contradiction"
                unsupported.append(f"{sent_clean} [{failure_reason}]")
            elif is_neg_conflict:
                status = "contradicted"
                effective_overlap = 0.0
                failure_reason = neg_err or "polarity contradiction"
                unsupported.append(f"{sent_clean} [{failure_reason}]")
            elif is_entity_conflict:
                status = "ungrounded"
                effective_overlap = min(effective_overlap, 0.15)
                failure_reason = ent_err or "swapped entity"
                unsupported.append(f"{sent_clean} [{failure_reason}]")
            elif is_pred_conflict:
                status = "ungrounded"
                effective_overlap = min(effective_overlap, 0.20)
                failure_reason = pred_err or "unsupported predicate"
                unsupported.append(f"{sent_clean} [{failure_reason}]")
            else:
                if effective_overlap >= 0.35:
                    status = "supported"
                    supported_count += 1
                elif effective_overlap >= 0.20:
                    status = "partially_supported"
                    unsupported.append(sent_clean)
                else:
                    status = "ungrounded"
                    unsupported.append(sent_clean)

            claim_details.append({
                "claim_id": idx,
                "statement": sent_clean,
                "status": status,
                "support_score": round(effective_overlap, 3),
                "evidence_span": best_span[:200],
                "failure_reason": failure_reason,
            })

        ratio = (supported_count / len(sentences)) if sentences else 1.0
        return {
            "claims_count": len(sentences),
            "supported_claims_count": supported_count,
            "factual_support_ratio": round(ratio, 3),
            "is_supported": ratio >= 0.50,
            "claim_details": claim_details,
            "unsupported_claims": unsupported,
        }

    def validate(
        self,
        generated_answer: str,
        sources: Sequence[CitationSource],
        supporting_chunks: Optional[Sequence[Any]] = None,
    ) -> CitationValidationReport:
        """
        Validate generated answer, evaluating:
        1. Factual claim grounding against retrieved contexts (citation-independent).
        2. Citation existence, validity, and alignment (if citations are present).
        """
        cited_ids = [int(m) for m in self.CITATION_PATTERN.findall(generated_answer)]
        source_id_map = {s.citation_id: s for s in sources}

        # Build combined context pool
        context_pool: List[Any] = list(sources)
        if supporting_chunks:
            context_pool.extend(supporting_chunks)

        source_texts = " ".join([s.snippet for s in sources])
        if supporting_chunks:
            source_texts += " " + " ".join([getattr(c, "text", "") for c in supporting_chunks])

        # Evaluate citation validity
        valid_citations: List[int] = []
        invalid_citations: List[int] = []
        document_verbatim_refs: List[int] = []

        for cid in cited_ids:
            if cid in source_id_map:
                valid_citations.append(cid)
            elif f"[{cid}]" in source_texts:
                document_verbatim_refs.append(cid)
            else:
                invalid_citations.append(cid)

        # 1. Evaluate Claim Support Independently of Citations
        claim_support_res = self.validate_claim_support(generated_answer, context_pool)
        factual_support_ratio = claim_support_res["factual_support_ratio"]
        unsupported = claim_support_res["unsupported_claims"]

        # 2. Build structured claim verifications with source mapping
        claim_verifications: List[ClaimVerification] = []
        aligned_citations = 0
        total_cited_claims = 0

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", generated_answer) if len(s.strip()) > 8]
        for idx, sent in enumerate(sentences, start=1):
            sent_citations = [int(m) for m in self.CITATION_PATTERN.findall(sent)]
            sent_clean = self.CITATION_PATTERN.sub("", sent).strip()
            content_words = self._extract_content_words(sent_clean)
            c_word_set = set(content_words)

            best_src_id: Optional[int] = sent_citations[0] if sent_citations else None
            best_support_score = 0.0
            best_span = ""
            best_doc_id = None
            best_chunk_id = None

            # Scan all sources for evidence span
            for src in sources:
                src_text = src.text or src.snippet
                src_sents = re.split(r"(?<=[.!?])\s+", src_text)
                for s_sent in src_sents:
                    src_words = set(re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", s_sent.lower()))
                    if c_word_set and src_words:
                        overlap = len(c_word_set & src_words) / len(c_word_set)
                        if overlap > best_support_score:
                            best_support_score = overlap
                            best_span = s_sent.strip()
                            best_src_id = src.citation_id
                            best_doc_id = src.document_id
                            best_chunk_id = src.chunk_id

            if best_support_score >= 0.35:
                status = "supported"
            elif best_support_score >= 0.20:
                status = "partially_supported"
            else:
                status = "ungrounded"

            if sent_citations:
                total_cited_claims += 1
                if any(c == best_src_id for c in sent_citations):
                    aligned_citations += 1

            claim_verifications.append(
                ClaimVerification(
                    claim_id=idx,
                    statement=sent_clean,
                    cited_source_id=best_src_id,
                    document_id=best_doc_id,
                    chunk_id=best_chunk_id,
                    status=status,
                    support_score=round(best_support_score, 3),
                    evidence_span=best_span[:200],
                )
            )

        has_citations = len(cited_ids) > 0
        citation_prec = (len(valid_citations) / len(cited_ids)) if cited_ids else 1.0
        citation_align = (aligned_citations / total_cited_claims) if total_cited_claims else 1.0

        is_factually_supported = factual_support_ratio >= 0.50

        # Overall validation: answer is valid if citations (if present) are non-hallucinated and claims are supported
        is_valid = len(invalid_citations) == 0 and (is_factually_supported or not sources)

        details = "All claims factually grounded in retrieved evidence."
        if invalid_citations:
            details = f"Detected hallucinated citation indices: {invalid_citations}"
            is_valid = False
        elif not is_factually_supported:
            details = f"Insufficient factual claim grounding: {factual_support_ratio:.1%} supported."
            is_valid = False
        elif not has_citations and sources:
            details = "Claims factually supported by context (citations omitted in text)."
        if invalid_citations:
            details = f"Detected hallucinated citation indices: {invalid_citations}"
            is_valid = False
        elif not is_factually_supported:
            details = f"Insufficient factual claim grounding: {factual_support_ratio:.1%} supported."
            is_valid = False
        elif not has_citations and sources:
            details = "Claims factually supported by context (citations omitted in text)."

        return CitationValidationReport(
            is_valid=is_valid,
            cited_ids=valid_citations,
            valid_citations=valid_citations,
            invalid_citations=invalid_citations,
            hallucinated_citations=invalid_citations,
            claim_verifications=claim_verifications,
            supported_claims_ratio=factual_support_ratio,
            factual_support_ratio=factual_support_ratio,
            is_factually_supported=is_factually_supported,
            has_citations=has_citations,
            citation_precision=round(citation_prec, 3),
            citation_alignment_score=round(citation_align, 3),
            unsupported_claims=unsupported,
            details=details,
        )
