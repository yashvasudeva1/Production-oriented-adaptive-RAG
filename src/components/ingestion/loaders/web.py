from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..base import BaseLoader
from ..exceptions import OptionalDependencyError
from ..models import DocumentBlock


class HTMLLoader(BaseLoader):
    name = "html-beautifulsoup"
    extensions = (".html", ".htm", ".xhtml")

    def load_blocks(self, path: Path) -> Iterable[DocumentBlock]:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise OptionalDependencyError(
                "HTML ingestion requires `beautifulsoup4`."
            ) from exc

        raw = path.read_bytes()
        soup = BeautifulSoup(raw, "html.parser")

        for tag in soup(["script", "style", "noscript", "template"]):
            tag.decompose()

        title = soup.title.get_text(" ", strip=True) if soup.title else None
        if title:
            yield DocumentBlock(
                block_id="",
                text=title,
                block_type="title",
                source_locator="html:title",
            )

        for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            text = element.get_text(" ", strip=True)
            if text:
                yield DocumentBlock(
                    block_id="",
                    text=text,
                    block_type="heading",
                    section=text,
                    source_locator=f"html:{element.name}",
                )

        for element in soup.find_all(["p", "li", "blockquote", "pre"]):
            text = element.get_text("\n", strip=True)
            if text:
                block_type = "code" if element.name == "pre" else "paragraph"
                yield DocumentBlock(
                    block_id="",
                    text=text,
                    block_type=block_type,
                    source_locator=f"html:{element.name}",
                )

        for table_index, table in enumerate(soup.find_all("table"), start=1):
            rows = []
            for tr in table.find_all("tr"):
                cells = tr.find_all(["th", "td"])
                if cells:
                    rows.append(" | ".join(c.get_text(" ", strip=True) for c in cells))

            text = "\n".join(rows).strip()
            if text:
                yield DocumentBlock(
                    block_id="",
                    text=text,
                    block_type="table",
                    source_locator=f"html:table:{table_index}",
                )
