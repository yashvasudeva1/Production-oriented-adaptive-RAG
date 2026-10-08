from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal


ChunkingStrategy = Literal[
    "auto",
    "recursive",
    "structural",
    "markdown",
    "code",
    "table",
    "transcript",
    "parent_child",
    "semantic",
]


@dataclass(slots=True)
class ChunkingConfig:
    """Configuration for chunking pipeline and specialized chunkers."""

    # Default size in approx tokens (or characters / 4)
    chunk_size: int = 200
    chunk_overlap: int = 30
    min_chunk_size: int = 30

    # Parent-child settings
    parent_chunk_size: int = 1200
    child_chunk_size: int = 250
    child_overlap: int = 40

    # Strategy
    strategy: ChunkingStrategy = "auto"
    preserve_hierarchy: bool = True

    # Tokenizer estimation factor: approx characters per token
    chars_per_token: float = 4.0

    # Separators for recursive splitting in order of precedence
    separators: List[str] = field(
        default_factory=lambda: [
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            "",
        ]
    )
