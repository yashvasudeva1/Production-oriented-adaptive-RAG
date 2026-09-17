from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import DocumentRecord

logger = logging.getLogger(__name__)

DEFAULT_STORE_PATH = (
    Path(__file__).resolve().parents[3] / "metadata" / "document_store.json"
)


class DocumentStore:
    """
    Manages document lifecycle, SHA-256 identity records,
    and parent-chunk full-text caches for context expansion.
    """

    def __init__(self, store_path: Path | str = DEFAULT_STORE_PATH) -> None:
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._documents: Dict[str, DocumentRecord] = {}
        self._parent_texts: Dict[str, str] = {}  # parent_id -> text
        self.load()

    def load(self) -> None:
        if not self.store_path.exists():
            return
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                docs = data.get("documents", {})
                for doc_id, doc_dict in docs.items():
                    self._documents[doc_id] = DocumentRecord(**doc_dict)
                self._parent_texts = data.get("parent_texts", {})
        except Exception as exc:
            logger.error(f"Error loading document store from {self.store_path}: {exc}")

    def save(self) -> None:
        try:
            payload = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "documents": {k: v.to_dict() for k, v in self._documents.items()},
                "parent_texts": self._parent_texts,
            }
            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.error(f"Error saving document store to {self.store_path}: {exc}")

    def get_by_sha256(self, sha256: str) -> Optional[DocumentRecord]:
        for doc in self._documents.values():
            if doc.sha256 == sha256 and doc.status == "INDEXED":
                return doc
        return None

    def get(self, document_id: str) -> Optional[DocumentRecord]:
        return self._documents.get(document_id)

    def list_documents(self) -> List[DocumentRecord]:
        return list(self._documents.values())

    def register_document(
        self,
        document_id: str,
        filename: str,
        sha256: str,
        size_bytes: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        status: str = "NEW",
    ) -> DocumentRecord:
        record = DocumentRecord(
            document_id=document_id,
            filename=filename,
            sha256=sha256,
            status=status,
            size_bytes=size_bytes,
            metadata=metadata or {},
        )
        self._documents[document_id] = record
        self.save()
        return record

    def update_status(
        self,
        document_id: str,
        status: str,
        chunk_count: int = 0,
        error_message: Optional[str] = None,
    ) -> None:
        if document_id in self._documents:
            doc = self._documents[document_id]
            doc.status = status
            if chunk_count:
                doc.chunk_count = chunk_count
            if error_message:
                doc.error_message = error_message
            if status == "INDEXED":
                doc.indexed_at = datetime.now(timezone.utc).isoformat()
            self.save()

    def store_parent_text(self, parent_id: str, text: str) -> None:
        self._parent_texts[parent_id] = text

    def get_parent_text(self, parent_id: str) -> Optional[str]:
        return self._parent_texts.get(parent_id)

    def delete(self, document_id: str) -> bool:
        if document_id in self._documents:
            del self._documents[document_id]
            # Clean up parent texts belonging to this document
            to_del = [pid for pid in self._parent_texts if pid.startswith(document_id)]
            for pid in to_del:
                del self._parent_texts[pid]
            self.save()
            return True
        return False
