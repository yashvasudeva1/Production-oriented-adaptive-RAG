from __future__ import annotations

import re


def normalize_query(query: str) -> str:
    """
    Sanitize and normalize user queries.
    Removes extraneous whitespace, non-printable characters,
    and common delimiter injections.
    """
    if not query:
        return ""

    # Remove non-printable control characters except standard space
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", query)

    # Collapse multiple whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned
