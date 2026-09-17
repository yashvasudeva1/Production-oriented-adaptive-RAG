from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass(slots=True)
class Chunk:
    """
    Unified representation of a text chunk with complete provenance.
    Preserves document, parent-child hierarchy, structural markers,
    and source coordinates.
    """
    chunk_id: str
    document_id: str
    text: str
    parent_id: Optional[str] = None
    chunk_type: str = "text"  # prose, code, table, slide, sheet, transcript, parent, child
    page: Optional[int] = None
    section: Optional[str] = None
    slide: Optional[int] = None
    sheet: Optional[str] = None
    source_locator: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Chunk:
        fields = {
            "chunk_id": data["chunk_id"],
            "document_id": data["document_id"],
            "text": data["text"],
            "parent_id": data.get("parent_id"),
            "chunk_type": data.get("chunk_type", "text"),
            "page": data.get("page"),
            "section": data.get("section"),
            "slide": data.get("slide"),
            "sheet": data.get("sheet"),
            "source_locator": data.get("source_locator"),
            "start_char": data.get("start_char"),
            "end_char": data.get("end_char"),
            "token_count": data.get("token_count", 0),
            "metadata": data.get("metadata", {}),
        }
        return cls(**fields)
