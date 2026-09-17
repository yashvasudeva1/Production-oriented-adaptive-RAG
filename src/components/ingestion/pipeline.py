from __future__ import annotations

import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Optional
from uuid import uuid4

from .config import IngestionConfig
from .exceptions import IngestionError
from .models import Document, DocumentBlock, IngestionResult
from .registry import LoaderRegistry
from .utils import detect_mime, make_block_id, normalize_text, safe_member_name, sha256_file


class IngestionPipeline:
    """
    Canonical entry point.

    Output is structured blocks, not retrieval chunks.
    Your chunking stage should consume Document.blocks later.
    """

    def __init__(
        self,
        config: Optional[IngestionConfig] = None,
        registry: Optional[LoaderRegistry] = None,
    ):
        self.config = config or IngestionConfig()
        self.registry = registry or LoaderRegistry(self.config)

    def ingest_file(self, path: str | Path) -> Document:
        path = Path(path)
        self.config.validate_path(path)
        return self._ingest_path(path)

    def ingest_many(
        self,
        paths: Iterable[str | Path],
        deduplicate: bool = True,
        strict: bool = False,
    ) -> IngestionResult:
        result = IngestionResult()
        seen_hashes: set[str] = set()

        for raw in paths:
            path = Path(raw)
            try:
                self.config.validate_path(path)
                digest = sha256_file(path)

                if deduplicate and digest in seen_hashes:
                    result.skipped.append({
                        "path": str(path),
                        "reason": "duplicate_sha256",
                    })
                    continue

                document = self._ingest_path(path)
                seen_hashes.add(digest)
                result.documents.append(document)

            except Exception as exc:
                if strict:
                    raise
                result.errors.append({
                    "path": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                })

        return result

    def _ingest_path(self, path: Path) -> Document:
        if self.config.recurse_archives and path.suffix.lower() in self.config.archive_suffixes:
            return self._ingest_archive(path)

        digest = sha256_file(path)
        loader = self.registry.get(path)

        raw_blocks = list(loader.load_blocks(path))
        blocks: list[DocumentBlock] = []

        for index, block in enumerate(raw_blocks):
            preserve = block.block_type == "code" and self.config.preserve_whitespace_in_code
            block.text = normalize_text(
                block.text,
                preserve_whitespace=preserve,
            )
            block.block_id = make_block_id(digest, index)

            if block.text:
                blocks.append(block)

        document = Document(
            document_id=f"doc_{digest[:20]}",
            source_path=str(path.resolve()),
            filename=path.name,
            extension=path.suffix.lower(),
            mime_type=detect_mime(path),
            sha256=digest,
            size_bytes=path.stat().st_size,
            metadata={
                "loader": loader.name,
                "ingested_with": "src.ingestion",
                **self.config.extra_metadata,
            },
            blocks=blocks,
        )

        return document

    def _ingest_archive(self, path: Path) -> Document:
        digest = sha256_file(path)
        member_blocks: list[DocumentBlock] = []

        with tempfile.TemporaryDirectory(prefix="rag_archive_") as tmp:
            tmp_root = Path(tmp)
            extracted = self._extract_archive(path, tmp_root)

            for member_path, virtual_name in extracted:
                try:
                    child = self._ingest_path(member_path)
                except Exception as exc:
                    member_blocks.append(DocumentBlock(
                        block_id="",
                        text=f"[Archive member failed: {virtual_name}] {exc}",
                        block_type="error",
                        source_locator=virtual_name,
                    ))
                    continue

                for block in child.blocks:
                    block.source_locator = f"{path.name}:{virtual_name}"
                    block.metadata["archive"] = path.name
                    member_blocks.append(block)

        for i, block in enumerate(member_blocks):
            block.block_id = make_block_id(digest, i)

        return Document(
            document_id=f"doc_{digest[:20]}",
            source_path=str(path.resolve()),
            filename=path.name,
            extension=path.suffix.lower(),
            mime_type=detect_mime(path),
            sha256=digest,
            size_bytes=path.stat().st_size,
            metadata={
                "loader": "archive-recursive",
                "archive_member_count": len(member_blocks),
                **self.config.extra_metadata,
            },
            blocks=member_blocks,
        )

    def _extract_archive(
        self,
        path: Path,
        target_dir: Path,
    ) -> list[tuple[Path, str]]:
        extracted: list[tuple[Path, str]] = []
        max_total = self.config.max_archive_total_size_mb * 1024 * 1024
        total = 0

        def accept_member(name: str, size: int) -> Optional[Path]:
            nonlocal total

            safe = safe_member_name(name)
            if safe is None:
                return None

            if len(extracted) >= self.config.max_archive_files:
                return None

            total += size
            if total > max_total:
                return None

            destination = target_dir / safe
            destination.parent.mkdir(parents=True, exist_ok=True)
            return destination

        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    destination = accept_member(info.filename, info.file_size)
                    if destination is None:
                        continue
                    with zf.open(info) as src, destination.open("wb") as dst:
                        dst.write(src.read())
                    extracted.append((destination, info.filename))
            return extracted

        if tarfile.is_tarfile(path):
            with tarfile.open(path, mode="r:*") as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    destination = accept_member(member.name, member.size)
                    if destination is None:
                        continue
                    src = tf.extractfile(member)
                    if src is None:
                        continue
                    with destination.open("wb") as dst:
                        dst.write(src.read())
                    extracted.append((destination, member.name))
            return extracted

        raise IngestionError(f"Unsupported archive container: {path.suffix}")
