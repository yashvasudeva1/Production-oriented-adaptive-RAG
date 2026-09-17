from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

from ..base import BaseLoader
from ..exceptions import OptionalDependencyError
from ..models import DocumentBlock


class JSONLoader(BaseLoader):
    name = "json-stdlib"
    extensions = (".json", ".jsonl", ".ndjson")

    def load_blocks(self, path: Path) -> Iterable[DocumentBlock]:
        text = path.read_text(encoding="utf-8", errors="replace")

        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            records = []
            for line in text.splitlines():
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        records.append(line)
            for index, record in enumerate(records, start=1):
                yield DocumentBlock(
                    block_id="",
                    text=json.dumps(record, ensure_ascii=False, indent=2),
                    block_type="json_record",
                    source_locator=f"record:{index}",
                )
            return

        data = json.loads(text)
        yield DocumentBlock(
            block_id="",
            text=json.dumps(data, ensure_ascii=False, indent=2),
            block_type="json",
            source_locator=str(path.name),
        )


class XMLLoader(BaseLoader):
    name = "xml-elementtree"
    extensions = (".xml", ".xhtml", ".rss", ".atom")

    def load_blocks(self, path: Path) -> Iterable[DocumentBlock]:
        tree = ET.parse(path)
        root = tree.getroot()

        # Human-readable, hierarchy-preserving-ish representation.
        lines: list[str] = []

        def visit(element: ET.Element, depth: int = 0) -> None:
            tag = element.tag.split("}")[-1]
            attrs = " ".join(f'{k}="{v}"' for k, v in element.attrib.items())
            prefix = "  " * depth

            header = f"{prefix}<{tag}{(' ' + attrs) if attrs else ''}>"
            lines.append(header)

            value = (element.text or "").strip()
            if value:
                lines.append(f"{prefix}  {value}")

            for child in element:
                visit(child, depth + 1)

            lines.append(f"{prefix}</{tag}>")

        visit(root)

        yield DocumentBlock(
            block_id="",
            text="\n".join(lines),
            block_type="xml",
            source_locator=str(path.name),
        )


class YAMLLoader(BaseLoader):
    name = "yaml-pyyaml"
    extensions = (".yaml", ".yml")

    def load_blocks(self, path: Path) -> Iterable[DocumentBlock]:
        try:
            import yaml
        except ImportError as exc:
            raise OptionalDependencyError(
                "YAML ingestion requires `pyyaml`."
            ) from exc

        data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
        yield DocumentBlock(
            block_id="",
            text=yaml.safe_dump(
                data,
                allow_unicode=True,
                sort_keys=False,
            ),
            block_type="yaml",
            source_locator=str(path.name),
        )
