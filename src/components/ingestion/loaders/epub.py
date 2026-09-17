from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..base import BaseLoader
from ..exceptions import OptionalDependencyError
from ..models import DocumentBlock


class EPUBLoader(BaseLoader):
    name = "epub-ebooklib"
    extensions = (".epub",)

    def load_blocks(self, path: Path) -> Iterable[DocumentBlock]:
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise OptionalDependencyError(
                "EPUB ingestion requires `ebooklib` and `beautifulsoup4`."
            ) from exc

        book = epub.read_epub(str(path))

        for index, item in enumerate(book.get_items_of_type(ebooklib.ITEM_DOCUMENT), start=1):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()

            text = soup.get_text("\n", strip=True)
            if text:
                yield DocumentBlock(
                    block_id="",
                    text=text,
                    block_type="epub_section",
                    source_locator=f"item:{index}",
                    metadata={"item_name": item.get_name()},
                )
