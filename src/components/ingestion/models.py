from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


@dataclass(slots=True)
class DocumentBlock:
    block_id: str
    text: str
    block_type: str = "text"
    page: Optional[int] = None
    section: Optional[str] = None
    sheet: Optional[str] = None
    slide: Optional[int] = None
    source_locator: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Document:
    document_id: str
    source_path: str
    filename: str
    extension: str
    mime_type: str
    sha256: str
    size_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    blocks: List[DocumentBlock] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(
            block.text.strip()
            for block in self.blocks
            if block.text and block.text.strip()
        )

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["text"] = self.text
        payload["char_count"] = self.char_count
        payload["word_count"] = self.word_count
        return payload

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass(slots=True)
class IngestionResult:
    documents: List[Document] = field(default_factory=list)
    skipped: List[Dict[str, str]] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)
