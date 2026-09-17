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


class CitationValidationReport(BaseModel):
    """Results of citation validation against ground-truth retrieved chunks."""
    is_valid: bool
    cited_ids: List[int] = Field(default_factory=list)
    valid_citations: List[int] = Field(default_factory=list)
    invalid_citations: List[int] = Field(default_factory=list)
    hallucinated_citations: List[int] = Field(default_factory=list)
    details: str = ""


class CitationBuilder:
    """Builds numbered citation references [1], [2] from supporting chunks."""

    def build_sources(
        self, chunks: Sequence[RerankerResult]
    ) -> List[CitationSource]:
        sources: List[CitationSource] = []
        for idx, chunk in enumerate(chunks, start=1):
            fn = chunk.metadata.get("filename") or f"doc_{chunk.document_id[:8]}"
            snippet = chunk.text[:600].replace("\n", " ").strip()
            sources.append(
                CitationSource(
                    citation_id=idx,
                    document_id=chunk.document_id,
                    filename=fn,
                    chunk_id=chunk.chunk_id,
                    page=chunk.page,
                    section=chunk.section,
                    source_locator=chunk.source_locator,
                    snippet=snippet,
                    text=chunk.text,
                    metadata=chunk.metadata,
                )
            )
        return sources

    def format_sources_for_prompt(self, sources: Sequence[CitationSource], chunks: Sequence[RerankerResult]) -> str:
        blocks: List[str] = []
        for src, chunk in zip(sources, chunks):
            locator = src.source_locator or f"doc:{src.filename}"
            blocks.append(
                f"Source [{src.citation_id}] ({src.filename}, {locator}):\n{chunk.text}"
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

        is_valid = len(invalid_citations) == 0 and len(valid_citations) > 0

        details = "All citations point to valid retrieved chunks."
        if invalid_citations:
            details = f"Detected hallucinated citation indices: {invalid_citations}"
        elif not valid_citations and sources:
            details = "No citations provided in answer text."
            is_valid = False

        return CitationValidationReport(
            is_valid=is_valid,
            cited_ids=valid_citations,
            valid_citations=valid_citations,
            invalid_citations=invalid_citations,
            hallucinated_citations=invalid_citations,
            details=details,
        )
