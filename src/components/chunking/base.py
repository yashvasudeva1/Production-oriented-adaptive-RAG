from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from typing import List, Sequence

from ..ingestion.models import DocumentBlock
from .config import ChunkingConfig
from .models import Chunk


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    @abstractmethod
    def chunk(self, blocks: Sequence[DocumentBlock], document_id: str) -> List[Chunk]:
        """Chunk a sequence of document blocks into a list of Chunks."""
        pass

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count quickly and robustly."""
        if not text:
            return 0
        words = len(text.split())
        chars = len(text)
        # Combine word count and character ratio for good approximation
        return max(1, max(words, int(chars / self.config.chars_per_token)))

    def make_chunk_id(self, document_id: str, identifier: str) -> str:
        """Generate a deterministic, reproducible chunk ID."""
        raw = f"{document_id}:{identifier}".encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()[:16]
        return f"chk_{digest}"
