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
    """Results of citation validation against ground-truth retrieved chunks."""
    is_valid: bool
    cited_ids: List[int] = Field(default_factory=list)
    valid_citations: List[int] = Field(default_factory=list)
    invalid_citations: List[int] = Field(default_factory=list)
    hallucinated_citations: List[int] = Field(default_factory=list)
    claim_verifications: List[ClaimVerification] = Field(default_factory=list)
    supported_claims_ratio: float = 1.0
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
            fn = meta.get("filename") or f"doc_{str(did)[:8]}"
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

    def validate(
        self,
        generated_answer: str,
        sources: Sequence[CitationSource],
        supporting_chunks: Optional[Sequence[Any]] = None,
    ) -> CitationValidationReport:
        found_matches = self.CITATION_PATTERN.findall(generated_answer)
        cited_ids = sorted(list({int(m) for m in found_matches}))
        source_id_map = {s.citation_id: s for s in sources}

        # Build corpus of verbatim source text to distinguish bibliography citations from hallucinations
        source_texts = " ".join([s.snippet for s in sources])
        if supporting_chunks:
            source_texts += " " + " ".join([getattr(c, "text", "") for c in supporting_chunks])

        valid_citations: List[int] = []
        invalid_citations: List[int] = []
        document_verbatim_refs: List[int] = []

        for cid in cited_ids:
            if cid in source_id_map:
                valid_citations.append(cid)
            elif f"[{cid}]" in source_texts:
                # The bracketed number was verbatim in the retrieved source text
                document_verbatim_refs.append(cid)
            else:
                invalid_citations.append(cid)

        # Claim-level evidence tracking
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", generated_answer) if len(s.strip()) > 8]
        claim_verifications: List[ClaimVerification] = []
        supported_claims = 0

        for idx, sent in enumerate(sentences, start=1):
            sent_citations = [int(m) for m in self.CITATION_PATTERN.findall(sent)]
            sent_clean = self.CITATION_PATTERN.sub("", sent).strip()
            s_words = set(re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", sent_clean.lower()))

            best_src_id: Optional[int] = sent_citations[0] if sent_citations else None
            best_support_score = 0.0
            best_span = ""
            best_doc_id = None
            best_chunk_id = None

            # Check cited source text first, or check all sources if uncited
            candidate_sources = [source_id_map[c] for c in sent_citations if c in source_id_map]
            if not candidate_sources and sources:
                candidate_sources = list(sources[:3])

            for src in candidate_sources:
                src_text = src.text or src.snippet
                src_sents = re.split(r"(?<=[.!?])\s+", src_text)
                for s_sent in src_sents:
                    src_words = set(re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", s_sent.lower()))
                    if s_words and src_words:
                        overlap = len(s_words & src_words) / len(s_words)
                        if overlap > best_support_score:
                            best_support_score = overlap
                            best_span = s_sent.strip()
                            best_src_id = src.citation_id
                            best_doc_id = src.document_id
                            best_chunk_id = src.chunk_id

            if best_support_score >= 0.35 or len(s_words) <= 3:
                status = "supported"
                supported_claims += 1
            elif best_support_score >= 0.18:
                status = "partially_supported"
                supported_claims += 1
            else:
                status = "ungrounded"

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

        supported_ratio = (supported_claims / len(sentences)) if sentences else 1.0

        is_valid = len(invalid_citations) == 0 and len(valid_citations) > 0 and supported_ratio >= 0.50

        details = "All citations point to valid retrieved chunks with verifiable claim grounding."
        if invalid_citations:
            details = f"Detected hallucinated citation indices: {invalid_citations}"
        elif not valid_citations and sources:
            details = "No citations provided in answer text."
            is_valid = False
        elif supported_ratio < 0.50:
            details = f"Insufficient claim-level grounding: {supported_ratio:.1%} supported."

        return CitationValidationReport(
            is_valid=is_valid,
            cited_ids=valid_citations,
            valid_citations=valid_citations,
            invalid_citations=invalid_citations,
            hallucinated_citations=invalid_citations,
            claim_verifications=claim_verifications,
            supported_claims_ratio=round(supported_ratio, 3),
            details=details,
        )
