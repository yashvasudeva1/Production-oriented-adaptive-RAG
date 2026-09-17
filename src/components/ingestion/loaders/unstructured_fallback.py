from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..base import BaseLoader
from ..exceptions import OptionalDependencyError
from ..models import DocumentBlock


class UnstructuredFallbackLoader(BaseLoader):
    """Last-resort parser for formats without a dedicated loader."""

    name = "unstructured-fallback"

    def can_load(self, path: Path) -> bool:
        return True

    def load_blocks(self, path: Path) -> Iterable[DocumentBlock]:
        try:
            from unstructured.partition.auto import partition
        except ImportError as exc:
            raise OptionalDependencyError(
                "Install `unstructured` to enable the broad fallback loader."
            ) from exc

        elements = partition(filename=str(path))

        for index, element in enumerate(elements, start=1):
            text = str(element).strip()
            if not text:
                continue

            metadata = {}
            try:
                metadata = element.metadata.to_dict()
            except Exception:
                pass

            category = getattr(element, "category", None)

            yield DocumentBlock(
                block_id="",
                text=text,
                block_type=(category.lower() if category else "unstructured"),
                source_locator=f"element:{index}",
                metadata=metadata,
            )
