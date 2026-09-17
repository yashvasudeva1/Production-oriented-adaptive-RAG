from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..base import BaseLoader
from ..exceptions import OptionalDependencyError
from ..models import DocumentBlock


class PDFLoader(BaseLoader):
    name = "pdf-pymupdf"
    extensions = (".pdf",)

    def load_blocks(self, path: Path) -> Iterable[DocumentBlock]:
        try:
            import pymupdf
        except ImportError as exc:
            raise OptionalDependencyError(
                "PDF ingestion requires `pymupdf`."
            ) from exc

        doc = pymupdf.open(path)

        try:
            for page_index, page in enumerate(doc, start=1):
                text = page.get_text("text", sort=self.config.pdf_sort_text).strip()

                # Scanned PDFs often return almost no text. OCR only when configured.
                if (
                    self.config.ocr_pdf_pages
                    and len(text) < self.config.ocr_pdf_min_chars
                ):
                    try:
                        text_page = page.get_textpage_ocr(
                            dpi=self.config.pdf_ocr_dpi,
                            full=True,
                        )
                        text = page.get_text(textpage=text_page).strip()
                    except Exception:
                        # OCR depends on an installed OCR backend (typically Tesseract).
                        pass

                if text:
                    yield DocumentBlock(
                        block_id="",
                        text=text,
                        block_type="page",
                        page=page_index,
                        source_locator=f"page:{page_index}",
                        metadata={
                            "pdf_page": page_index,
                            "ocr_attempted": (
                                self.config.ocr_pdf_pages
                                and len(text) < self.config.ocr_pdf_min_chars
                            ),
                        },
                    )
        finally:
            doc.close()
