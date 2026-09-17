"""
ResearchLens ingestion component test suite.

Project layout expected:

    production rag/
    ├── src/
    │   ├── __init__.py
    │   └── components/
    │       ├── __init__.py
    │       └── ingestion/
    │           ├── __init__.py
    │           └── ...

Run from the project root:

    python test_ingestion.py

Useful options:

    python test_ingestion.py --keep-fixtures
    python test_ingestion.py --test-media

Design goals:
- Use src.components.ingestion for the user's current project structure.
- Fall back to src.ingestion for portability.
- Treat missing optional dependencies as SKIP, not FAIL.
- Test structure/provenance, not just "some text came out".
- Test archive recursion and SHA-256 deduplication.
- Never download/run Whisper unless --test-media is explicitly passed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Callable

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# CLI

parser = argparse.ArgumentParser(
    description="ResearchLens ingestion component test suite"
)
parser.add_argument(
    "--keep-fixtures",
    action="store_true",
    help="Keep generated fixture files for manual inspection.",
)
parser.add_argument(
    "--test-media",
    action="store_true",
    help=(
        "Actually run faster-whisper. Requires "
        "test-fixtures/voice-test.wav and may download a model."
    ),
)
args = parser.parse_args()


# Project import setup

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_ingestion_package():
    """
    Prefer the user's current layout:

        src.components.ingestion

    Keep a fallback for portability if someone uses:

        src.ingestion
    """
    candidates = (
        "src.components.ingestion",
        "src.ingestion",
    )

    last_error: Exception | None = None

    for package_name in candidates:
        try:
            package = importlib.import_module(package_name)
            return package_name, package
        except ModuleNotFoundError as exc:
            last_error = exc

    raise ModuleNotFoundError(
        "Could not import either `src.components.ingestion` "
        "or `src.ingestion`. "
        "Check that the project root is correct and that "
        "`src/__init__.py`, `src/components/__init__.py`, and "
        "the ingestion package exist."
    ) from last_error


# Import the main package once. Optional third-party dependencies are loaded
# by the individual loaders lazily, so this should not require every package.
INGESTION_PACKAGE, INGESTION = load_ingestion_package()

IngestionConfig = INGESTION.IngestionConfig
IngestionPipeline = INGESTION.IngestionPipeline


def ingestion_module(module_name: str):
    return importlib.import_module(f"{INGESTION_PACKAGE}.{module_name}")


def ingestion_model(name: str):
    return getattr(ingestion_module("models"), name)


def ingestion_exception(name: str):
    return getattr(ingestion_module("exceptions"), name)


OptionalDependencyError = ingestion_exception("OptionalDependencyError")
UnsupportedFileTypeError = ingestion_exception("UnsupportedFileTypeError")


# Test state

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))

    symbols = {
        PASS: "✓",
        FAIL: "✗",
        SKIP: "○",
    }

    suffix = f" — {detail}" if detail else ""
    print(f"{symbols[status]} {name}{suffix}")


def run_test(name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
    except OptionalDependencyError as exc:
        record(name, SKIP, str(exc))
    except ImportError as exc:
        record(name, SKIP, f"Optional dependency unavailable: {exc}")
    except Exception as exc:
        record(name, FAIL, f"{type(exc).__name__}: {exc}")
    else:
        record(name, PASS)


# Fixture generation

def create_fixtures(root: Path) -> dict[str, Path]:
    """
    Create representative files for every loader we can reasonably test
    without requiring external downloads.
    """
    files: dict[str, Path] = {}

    # Plain text
    p = root / "sample.txt"
    p.write_text(
        "ResearchLens ingestion test.\n"
        "This is a plain text document.\n"
        "It contains multiple lines.\n",
        encoding="utf-8",
    )
    files["txt"] = p

    # Markdown
    p = root / "sample.md"
    p.write_text(
        "# ResearchLens\n\n"
        "## Architecture\n\n"
        "This is a **Markdown** document.\n\n"
        "- Dense retrieval\n"
        "- Hybrid retrieval\n"
        "- Reranking\n\n"
        "```python\n"
        "def hello():\n"
        "    return 'world'\n"
        "```\n",
        encoding="utf-8",
    )
    files["md"] = p

    # Python
    p = root / "sample.py"
    p.write_text(
        "def fibonacci(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return fibonacci(n - 1) + fibonacci(n - 2)\n",
        encoding="utf-8",
    )
    files["py"] = p

    # JSON / JSONL
    p = root / "sample.json"
    p.write_text(
        json.dumps(
            {
                "project": "ResearchLens",
                "version": 1,
                "features": ["RAG", "reranking"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    files["json"] = p

    p = root / "sample.jsonl"
    p.write_text(
        '{"id": 1, "text": "first"}\n'
        '{"id": 2, "text": "second"}\n',
        encoding="utf-8",
    )
    files["jsonl"] = p

    # XML / YAML

    p = root / "sample.xml"
    p.write_text(
        '<?xml version="1.0"?>'
        "<root>"
        "<title>ResearchLens</title>"
        "<item><name>RAG</name></item>"
        "</root>",
        encoding="utf-8",
    )
    files["xml"] = p

    p = root / "sample.yaml"
    p.write_text(
        "project: ResearchLens\n"
        "version: 1\n"
        "features:\n"
        "  - ingestion\n"
        "  - retrieval\n",
        encoding="utf-8",
    )
    files["yaml"] = p

    # CSV / TSV
    p = root / "sample.csv"
    p.write_text(
        "id,name,score\n"
        "1,alpha,0.91\n"
        "2,beta,0.87\n",
        encoding="utf-8",
    )
    files["csv"] = p

    p = root / "sample.tsv"
    p.write_text(
        "id\tname\tscore\n"
        "1\talpha\t0.91\n"
        "2\tbeta\t0.87\n",
        encoding="utf-8",
    )
    files["tsv"] = p

    # HTML
    p = root / "sample.html"
    p.write_text(
        """<!doctype html>
<html>
<head>
    <title>ResearchLens Test</title>
    <style>body { font-family: sans-serif; }</style>
</head>
<body>
    <h1>Architecture</h1>
    <h2>Retrieval</h2>
    <p>This is HTML content.</p>

    <ul>
        <li>Dense retrieval</li>
        <li>Hybrid retrieval</li>
    </ul>

    <table>
        <tr><th>Component</th><th>Status</th></tr>
        <tr><td>Qdrant</td><td>Ready</td></tr>
    </table>

    <script>console.log("THIS MUST NOT BE INGESTED");</script>
</body>
</html>
""",
        encoding="utf-8",
    )
    files["html"] = p

    # EML
    p = root / "sample.eml"
    p.write_text(
        """From: sender@example.com
To: receiver@example.com
Subject: ResearchLens ingestion test
Date: Tue, 17 Sep 2026 10:00:00 +0530
Content-Type: text/plain; charset="utf-8"

Hello from the ResearchLens ingestion test.
""",
        encoding="utf-8",
    )
    files["eml"] = p

    # DOCX
    try:
        from docx import Document as WordDocument
    except ImportError:
        pass
    else:
        p = root / "sample.docx"

        doc = WordDocument()
        doc.add_heading("ResearchLens", level=1)
        doc.add_heading("Architecture", level=2)
        doc.add_paragraph("This is a DOCX paragraph.")

        table = doc.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Component"
        table.cell(0, 1).text = "Status"
        table.cell(1, 0).text = "Ingestion"
        table.cell(1, 1).text = "Testing"

        doc.save(p)
        files["docx"] = p

    # PPTX
    try:
        from pptx import Presentation
        from pptx.util import Inches
    except ImportError:
        pass
    else:
        p = root / "sample.pptx"

        prs = Presentation()

        slide = prs.slides.add_slide(prs.slide_layouts[5])
        slide.shapes.title.text = "ResearchLens"

        textbox = slide.shapes.add_textbox(
            Inches(1),
            Inches(2),
            Inches(8),
            Inches(1),
        )
        textbox.text_frame.text = "PPTX ingestion test"

        prs.save(p)
        files["pptx"] = p

    # XLSX
    try:
        from openpyxl import Workbook
    except ImportError:
        pass
    else:
        p = root / "sample.xlsx"

        wb = Workbook()
        ws = wb.active
        ws.title = "Retrieval"
        ws.append(["Component", "Status"])
        ws.append(["Qdrant", "Ready"])
        ws.append(["Reranker", "Ready"])
        wb.save(p)

        files["xlsx"] = p

    # ODT
    try:
        from odf.opendocument import OpenDocumentText
        from odf.text import P
    except ImportError:
        pass
    else:
        p = root / "sample.odt"

        odt = OpenDocumentText()
        odt.text.addElement(P(text="ResearchLens"))
        odt.text.addElement(P(text="ODT ingestion test."))
        odt.save(str(p))

        files["odt"] = p

    # PDF
    try:
        import pymupdf
    except ImportError:
        pass
    else:
        p = root / "sample.pdf"

        pdf = pymupdf.open()
        page = pdf.new_page()
        page.insert_text(
            (72, 72),
            "ResearchLens PDF ingestion test.",
        )
        page.insert_text(
            (72, 100),
            "Page 1 contains searchable text.",
        )
        pdf.save(str(p))
        pdf.close()

        files["pdf"] = p

    # Image
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        pass
    else:
        p = root / "sample.png"

        image = Image.new("RGB", (1600, 500), "white")
        draw = ImageDraw.Draw(image)

        # Large high-contrast text makes OCR smoke testing less dependent
        # on default font rendering.
        draw.text(
            (60, 180),
            "RESEARCHLENS OCR TEST",
            fill="black",
        )

        image.save(p)
        files["image"] = p

    # Archive containing supported files

    archive = root / "sample.zip"

    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        zf.write(files["txt"], arcname="nested/sample.txt")
        zf.write(files["json"], arcname="nested/sample.json")

    files["zip"] = archive

    return files


# Assertions

def assert_document(
    document,
    expected_extension: str | None = None,
) -> None:
    assert document.document_id.startswith("doc_"), (
        f"Invalid document_id: {document.document_id}"
    )

    assert document.filename, "filename missing"

    assert document.sha256, "SHA-256 missing"
    assert len(document.sha256) == 64, (
        f"Invalid SHA-256 length: {len(document.sha256)}"
    )

    assert document.size_bytes > 0, "input file is empty"

    assert document.blocks, "no DocumentBlock objects were produced"

    assert document.text.strip(), "document text is empty"

    if expected_extension:
        assert document.extension == expected_extension, (
            f"Expected extension {expected_extension}, "
            f"got {document.extension}"
        )

    block_ids = [
        block.block_id
        for block in document.blocks
    ]

    assert len(block_ids) == len(set(block_ids)), (
        "Duplicate block IDs detected"
    )

    for block in document.blocks:
        assert block.block_id, "block_id missing"
        assert block.text.strip(), (
            f"Empty block: {block.block_id}"
        )
        assert block.block_type, (
            f"block_type missing: {block.block_id}"
        )
        assert block.source_locator is not None, (
            f"source_locator missing: {block.block_id}"
        )


# Pipeline factory

def make_pipeline() -> IngestionPipeline:
    return IngestionPipeline(
        IngestionConfig(
            ocr_images=True,
            # Normal PDF smoke test should use searchable text and should not
            # invoke an external OCR binary.
            ocr_pdf_pages=False,
            transcribe_audio_video=False,
        )
    )


# Architecture tests

def test_imports() -> None:
    assert INGESTION is not None
    assert IngestionConfig is not None
    assert IngestionPipeline is not None


def test_registry() -> None:
    registry_module = ingestion_module("registry")
    LoaderRegistry = registry_module.LoaderRegistry

    registry = LoaderRegistry(
        IngestionConfig(
            ocr_images=False,
            ocr_pdf_pages=False,
        )
    )

    names = [
        loader.name
        for loader in registry.all()
    ]

    required = {
        "pdf-pymupdf",
        "docx-python-docx",
        "pptx-python-pptx",
        "spreadsheet-openpyxl",
        "csv-stdlib",
        "html-beautifulsoup",
        "json-stdlib",
        "xml-elementtree",
        "yaml-pyyaml",
        "image-pillow-tesseract",
        "eml-email-stdlib",
        "msg-extract-msg",
        "epub-ebooklib",
        "media-faster-whisper",
        "markdown",
        "source-code",
        "plain-text",
        "unstructured-fallback",
    }

    missing = required - set(names)

    assert not missing, (
        f"Missing registered loaders: {sorted(missing)}"
    )

    assert names[-1] == "unstructured-fallback", (
        "Unstructured fallback must be registered last"
    )


def test_config_and_models() -> None:
    config = IngestionConfig(
        max_file_size_mb=1,
        ocr_images=False,
        ocr_pdf_pages=False,
    )

    assert config.max_file_size_mb == 1
    assert config.archive_suffixes

    Document = ingestion_model("Document")
    DocumentBlock = ingestion_model("DocumentBlock")

    block = DocumentBlock(
        block_id="b1",
        text="hello",
        block_type="paragraph",
        page=2,
        section="Intro",
        source_locator="page:2",
    )

    document = Document(
        document_id="doc_test",
        source_path="test.txt",
        filename="test.txt",
        extension=".txt",
        mime_type="text/plain",
        sha256="0" * 64,
        size_bytes=5,
        blocks=[block],
    )

    assert document.text == "hello"
    assert document.word_count == 1
    assert document.char_count == 5


# Generic loader test

def test_generic_file(
    files: dict[str, Path],
    label: str,
) -> None:
    path = files[label]

    document = make_pipeline().ingest_file(path)

    assert_document(
        document,
        path.suffix.lower(),
    )

    assert document.metadata.get("loader"), (
        "document.metadata['loader'] is missing"
    )


# Specialized tests

def test_markdown_structure(files: dict[str, Path]) -> None:
    document = make_pipeline().ingest_file(files["md"])

    assert_document(document, ".md")

    types = {
        block.block_type
        for block in document.blocks
    }

    assert "heading" in types
    assert "code" in types


def test_pdf_provenance(files: dict[str, Path]) -> None:
    if "pdf" not in files:
        raise OptionalDependencyError(
            "pymupdf is not installed."
        )

    document = make_pipeline().ingest_file(files["pdf"])

    assert_document(document, ".pdf")

    page_blocks = [
        block
        for block in document.blocks
        if block.block_type == "page"
    ]

    assert page_blocks, "No PDF page blocks produced"

    assert all(
        block.page is not None
        for block in page_blocks
    )


def test_docx_structure(files: dict[str, Path]) -> None:
    if "docx" not in files:
        raise OptionalDependencyError(
            "python-docx is not installed."
        )

    document = make_pipeline().ingest_file(files["docx"])

    assert_document(document, ".docx")

    types = {
        block.block_type
        for block in document.blocks
    }

    assert "heading" in types
    assert "table" in types


def test_pptx_structure(files: dict[str, Path]) -> None:
    if "pptx" not in files:
        raise OptionalDependencyError(
            "python-pptx is not installed."
        )

    document = make_pipeline().ingest_file(files["pptx"])

    assert_document(document, ".pptx")

    slide_blocks = [
        block
        for block in document.blocks
        if block.block_type == "slide"
    ]

    assert slide_blocks, "No PPTX slide blocks produced"

    assert all(
        block.slide is not None
        for block in slide_blocks
    )


def test_xlsx_structure(files: dict[str, Path]) -> None:
    if "xlsx" not in files:
        raise OptionalDependencyError(
            "openpyxl is not installed."
        )

    document = make_pipeline().ingest_file(files["xlsx"])

    assert_document(document, ".xlsx")

    assert any(
        block.sheet == "Retrieval"
        for block in document.blocks
    ), "Expected Retrieval sheet provenance"


def test_html_structure(files: dict[str, Path]) -> None:
    try:
        import bs4  # noqa: F401
    except ImportError:
        raise OptionalDependencyError(
            "beautifulsoup4 is not installed."
        )

    document = make_pipeline().ingest_file(files["html"])

    assert_document(document, ".html")

    joined = document.text.lower()

    assert "researchlens" in joined
    assert "qdrant" in joined

    # Script/style content should have been removed.
    assert "this must not be ingested" not in joined


def test_image_ocr(files: dict[str, Path]) -> None:
    if "image" not in files:
        raise OptionalDependencyError(
            "Pillow is not installed."
        )

    try:
        import pytesseract
    except ImportError:
        raise OptionalDependencyError(
            "pytesseract is not installed."
        )

    cmd = IngestionConfig().tesseract_cmd
    if cmd and Path(cmd).exists():
        pytesseract.pytesseract.tesseract_cmd = cmd

    # Distinguish "Python package missing" from "Tesseract executable missing".
    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        raise OptionalDependencyError(
            "Tesseract executable is not installed or is not on PATH."
        ) from exc

    document = make_pipeline().ingest_file(files["image"])

    assert_document(document, ".png")

    normalized = re.sub(
        r"[^a-z0-9]",
        "",
        document.text.lower(),
    )

    assert "researchlens" in normalized, (
        f"OCR did not detect expected text: {document.text!r}"
    )


def test_archive(files: dict[str, Path]) -> None:
    document = make_pipeline().ingest_file(files["zip"])

    assert_document(document, ".zip")

    assert document.metadata.get("loader") == "archive-recursive"

    # The current implementation calls this "archive_member_count" but it
    # counts extracted blocks. We therefore verify that nested content came
    # through rather than assuming the metadata represents physical members.
    assert document.metadata.get("archive_member_count", 0) >= 2

    joined = document.text.lower()

    assert "researchlens ingestion test" in joined
    assert '"project": "researchlens"' in joined


def test_batch_deduplication(
    files: dict[str, Path],
) -> None:
    pipeline = make_pipeline()

    candidates = [
        files["txt"],
        files["json"],
        files["txt"],      # deliberate duplicate
    ]

    # HTML is deliberately NOT included here. Optional dependency availability
    # should not determine whether the deduplication test passes.
    result = pipeline.ingest_many(
        candidates,
        deduplicate=True,
        strict=False,
    )

    assert len(result.documents) == 2, (
        f"Expected 2 unique documents, got {len(result.documents)}"
    )

    assert len(result.skipped) == 1

    assert result.skipped[0]["reason"] == "duplicate_sha256"

    assert not result.errors, (
        f"Unexpected batch errors: {result.errors}"
    )


def test_batch_with_optional_html(
    files: dict[str, Path],
) -> None:
    """
    Separate test for batch behavior when an optional dependency is absent.

    This verifies that one problematic optional format does not prevent other
    documents from being ingested.
    """
    pipeline = make_pipeline()

    candidates = [
        files["txt"],
        files["json"],
        files["html"],
    ]

    result = pipeline.ingest_many(
        candidates,
        deduplicate=True,
        strict=False,
    )

    assert len(result.documents) >= 2, (
        f"Expected at least TXT + JSON to succeed, "
        f"got {len(result.documents)}"
    )

    # Any HTML failure must not cause TXT/JSON failures.
    non_html_errors = [
        error
        for error in result.errors
        if not error["path"].lower().endswith(".html")
    ]

    assert not non_html_errors, (
        f"Unexpected non-HTML errors: {non_html_errors}"
    )


def test_unsupported_behavior(files: dict[str, Path]) -> None:
    """
    With the Unstructured fallback installed, an arbitrary extension may be
    routed there. Without it, an informative OptionalDependencyError or
    UnsupportedFileTypeError is acceptable.
    """
    path = files["txt"].with_name("sample.unsupported")

    path.write_text(
        "unsupported format test",
        encoding="utf-8",
    )

    try:
        document = make_pipeline().ingest_file(path)

    except OptionalDependencyError as exc:
        assert "unstructured" in str(exc).lower()

    except UnsupportedFileTypeError:
        # Acceptable if the fallback has been intentionally omitted/disabled.
        return

    else:
        assert_document(document)
        assert document.blocks


def test_media_registration() -> None:
    registry_module = ingestion_module("registry")
    LoaderRegistry = registry_module.LoaderRegistry

    registry = LoaderRegistry(
        IngestionConfig(
            transcribe_audio_video=False,
        )
    )

    names = [
        loader.name
        for loader in registry.all()
    ]

    assert "media-faster-whisper" in names


def test_media_actual() -> None:
    if not args.test_media:
        raise OptionalDependencyError(
            "Skipped by default. Use --test-media to run Whisper."
        )

    media = PROJECT_ROOT / "test-fixtures" / "voice-test.wav"

    if not media.exists():
        raise OptionalDependencyError(
            "Create test-fixtures/voice-test.wav first."
        )

    config = IngestionConfig(
        transcribe_audio_video=True,
        whisper_model="base",
        whisper_device="auto",
    )

    pipeline = IngestionPipeline(config)
    document = pipeline.ingest_file(media)

    assert_document(document, ".wav")

    assert any(
        block.block_type == "transcript"
        for block in document.blocks
    ), "No transcript blocks produced"


# Main

def main() -> int:
    print("-" * 72)
    print("ResearchLens — Ingestion Component Test Suite")
    print(f"Project root:     {PROJECT_ROOT}")
    print(f"Ingestion import: {INGESTION_PACKAGE}")
    print("-" * 72)
    print()

    fixture_dir = PROJECT_ROOT / ".ingestion_test_fixtures"

    if fixture_dir.exists():
        shutil.rmtree(fixture_dir)

    fixture_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = create_fixtures(fixture_dir)

    print(f"Created {len(files)} fixture files.")
    print()

    # Package architecture
    run_test(
        "Package imports",
        test_imports,
    )

    run_test(
        "Loader registry",
        test_registry,
    )

    run_test(
        "Config + models contract",
        test_config_and_models,
    )

    # Core formats
    generic_formats = (
        "txt",
        "json",
        "jsonl",
        "xml",
        "yaml",
        "csv",
        "tsv",
        "html",
        "eml",
        "py",
    )

    for label in generic_formats:
        if label not in files:
            continue

        run_test(
            f"{label.upper()} ingestion",
            lambda label=label: test_generic_file(
                files,
                label,
            ),
        )

    # Structure-sensitive formats
    run_test(
        "Markdown structure + code blocks",
        lambda: test_markdown_structure(files),
    )

    run_test(
        "PDF page provenance",
        lambda: test_pdf_provenance(files),
    )

    run_test(
        "DOCX paragraph/table structure",
        lambda: test_docx_structure(files),
    )

    run_test(
        "PPTX slide provenance",
        lambda: test_pptx_structure(files),
    )

    run_test(
        "XLSX sheet/table structure",
        lambda: test_xlsx_structure(files),
    )

    run_test(
        "Image OCR",
        lambda: test_image_ocr(files),
    )

    # Cross-cutting behavior
    run_test(
        "Recursive archive ingestion",
        lambda: test_archive(files),
    )

    run_test(
        "Batch ingestion + SHA-256 deduplication",
        lambda: test_batch_deduplication(files),
    )

    run_test(
        "Batch survives optional HTML failure",
        lambda: test_batch_with_optional_html(files),
    )

    run_test(
        "Unsupported/fallback behavior",
        lambda: test_unsupported_behavior(files),
    )

    run_test(
        "Media loader registration",
        test_media_registration,
    )

    run_test(
        "Actual media transcription",
        test_media_actual,
    )

    # Summary
    print()
    print("-" * 72)
    print("SUMMARY")
    print("-" * 72)

    counts = {
        PASS: sum(status == PASS for _, status, _ in results),
        FAIL: sum(status == FAIL for _, status, _ in results),
        SKIP: sum(status == SKIP for _, status, _ in results),
    }

    print(f"PASS: {counts[PASS]}")
    print(f"FAIL: {counts[FAIL]}")
    print(f"SKIP: {counts[SKIP]}")

    failures = [
        (name, detail)
        for name, status, detail in results
        if status == FAIL
    ]

    if failures:
        print("\nFailures:")
        for name, detail in failures:
            print(f"  - {name}: {detail}")

    print()

    if args.keep_fixtures:
        print(f"Fixtures retained at: {fixture_dir}")
    else:
        shutil.rmtree(
            fixture_dir,
            ignore_errors=True,
        )

    # FAIL only determines the process exit code.
    # SKIP is expected when optional packages/system tools are not installed.
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
