from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..base import BaseLoader
from ..exceptions import OptionalDependencyError
from ..models import DocumentBlock


class ImageOCRLoader(BaseLoader):
    name = "image-pillow-tesseract"
    extensions = (
        ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff",
        ".gif",
    )

    def load_blocks(self, path: Path) -> Iterable[DocumentBlock]:
        if not self.config.ocr_images:
            return

        try:
            from PIL import Image
        except ImportError as exc:
            raise OptionalDependencyError(
                "Image ingestion requires `pillow`."
            ) from exc

        try:
            import pytesseract
        except ImportError as exc:
            raise OptionalDependencyError(
                "Image OCR requires `pytesseract` plus a system Tesseract installation."
            ) from exc

        if getattr(self.config, "tesseract_cmd", None):
            pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_cmd

        image = Image.open(path)
        text = pytesseract.image_to_string(image)

        if text.strip():
            yield DocumentBlock(
                block_id="",
                text=text,
                block_type="ocr",
                source_locator=str(path.name),
                metadata={
                    "width": image.width,
                    "height": image.height,
                    "ocr": True,
                },
            )
