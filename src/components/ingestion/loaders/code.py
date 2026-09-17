from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..base import BaseLoader
from ..models import DocumentBlock


CODE_EXTENSIONS = (
    ".py", ".pyw", ".js", ".jsx", ".ts", ".tsx",
    ".java", ".kt", ".kts", ".go", ".rs", ".c", ".h",
    ".cpp", ".cc", ".cxx", ".hpp", ".cs", ".php", ".rb",
    ".swift", ".scala", ".sh", ".bash", ".zsh", ".fish",
    ".ps1", ".sql", ".r", ".lua", ".dart", ".m", ".mm",
    ".css", ".scss", ".sass", ".less",
    ".vue", ".svelte", ".graphql", ".gql",
    ".dockerfile", ".proto", ".toml", ".ini", ".cfg",
)


class CodeLoader(BaseLoader):
    name = "source-code"
    extensions = CODE_EXTENSIONS

    def can_load(self, path: Path) -> bool:
        suffix = path.suffix.lower()
        name = path.name.lower()

        return (
            suffix in self.extensions
            or name in {"dockerfile", "makefile", "jenkinsfile"}
        )

    def load_blocks(self, path: Path) -> Iterable[DocumentBlock]:
        text = path.read_text(encoding="utf-8", errors="replace")
        yield DocumentBlock(
            block_id="",
            text=text,
            block_type="code",
            source_locator=str(path.name),
            metadata={"language_hint": _language_hint(path)},
        )


def _language_hint(path: Path) -> str:
    return {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".cpp": "cpp",
        ".c": "c",
        ".sql": "sql",
        ".sh": "shell",
        ".ps1": "powershell",
        ".html": "html",
        ".css": "css",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
    }.get(path.suffix.lower(), path.suffix.lower().lstrip("."))


class _Noop:
    pass
