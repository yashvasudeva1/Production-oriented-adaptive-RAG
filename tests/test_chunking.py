from __future__ import annotations

import pytest
from src.components.chunking import (
    Chunk,
    ChunkingConfig,
    ChunkingPipeline,
    CodeChunker,
    MarkdownChunker,
    ParentChildChunker,
    RecursiveChunker,
    StructuralChunker,
    TableChunker,
    TranscriptChunker,
)
from src.components.ingestion.models import Document, DocumentBlock


def test_recursive_chunker_provenance():
    chunker = RecursiveChunker(ChunkingConfig(chunk_size=50, chunk_overlap=10))
    blocks = [
        DocumentBlock(
            block_id="b1",
            text="First sentence. Second sentence with more text. Third sentence with even more text.",
            page=1,
            section="Introduction",
            source_locator="page:1",
        )
    ]
    chunks = chunker.chunk(blocks, "doc_test")
    assert len(chunks) >= 1
    for c in chunks:
        assert c.document_id == "doc_test"
        assert c.page == 1
        assert c.section == "Introduction"
        assert c.source_locator == "page:1"
        assert c.token_count > 0


def test_structural_chunker_heading_preservation():
    chunker = StructuralChunker(ChunkingConfig(chunk_size=40))
    blocks = [
        DocumentBlock(
            block_id="b1",
            text="Paragraph one of the section.\n\nParagraph two of the section with more details.",
            section="Architecture Overview",
            source_locator="section:Architecture",
        )
    ]
    chunks = chunker.chunk(blocks, "doc_struct")
    assert len(chunks) >= 1
    assert "Architecture Overview" in chunks[0].text
    assert chunks[0].section == "Architecture Overview"


def test_markdown_chunker_hierarchy():
    chunker = MarkdownChunker()
    blocks = [
        DocumentBlock(
            block_id="b1",
            text="# Title\nIntro text.\n\n## Details\nHere are details.\n```python\nx = 1\n```",
            block_type="markdown",
        )
    ]
    chunks = chunker.chunk(blocks, "doc_md")
    assert len(chunks) >= 1
    assert any("Title > Details" in c.section for c in chunks if c.section)


def test_code_chunker_function_scope():
    chunker = CodeChunker(ChunkingConfig(chunk_size=100))
    code_text = (
        "def process_data(items):\n"
        "    \"\"\"Docstring.\"\"\"\n"
        "    return [x * 2 for x in items]\n\n"
        "class ModelTrainer:\n"
        "    def __init__(self):\n"
        "        pass\n"
    )
    blocks = [
        DocumentBlock(
            block_id="b1",
            text=code_text,
            block_type="code",
            source_locator="src/main.py",
        )
    ]
    chunks = chunker.chunk(blocks, "doc_code")
    assert len(chunks) >= 2
    assert any("def process_data" in c.text for c in chunks)
    assert any("class ModelTrainer" in c.text for c in chunks)


def test_table_chunker_repeated_headers():
    chunker = TableChunker()
    table_csv = (
        "Name,Age,Role\n"
        + "\n".join(f"Person_{i},{20+i},Engineer" for i in range(40))
    )
    blocks = [
        DocumentBlock(
            block_id="b1",
            text=table_csv,
            block_type="table",
            sheet="Employees",
        )
    ]
    chunks = chunker.chunk(blocks, "doc_tbl")
    assert len(chunks) >= 2
    for c in chunks:
        assert "Name,Age,Role" in c.text
        assert c.sheet == "Employees"


def test_parent_child_chunker():
    chunker = ParentChildChunker(
        ChunkingConfig(parent_chunk_size=100, child_chunk_size=30, child_overlap=5)
    )
    long_text = "\n\n".join(
        f"Paragraph {i}: Detailed explanation of component {i} in the pipeline."
        for i in range(10)
    )
    blocks = [
        DocumentBlock(
            block_id="b1",
            text=long_text,
            page=2,
            source_locator="page:2",
        )
    ]
    chunks = chunker.chunk(blocks, "doc_pc")
    parents = [c for c in chunks if c.chunk_type == "parent"]
    children = [c for c in chunks if c.chunk_type == "child"]

    assert len(parents) >= 1
    assert len(children) >= 2
    for child in children:
        assert child.parent_id is not None
        assert any(child.parent_id == p.chunk_id for p in parents)


def test_chunking_pipeline_validation():
    pipeline = ChunkingPipeline()
    doc = Document(
        document_id="doc_pipe",
        source_path="sample.txt",
        filename="sample.txt",
        extension=".txt",
        mime_type="text/plain",
        sha256="abc12345",
        size_bytes=50,
        blocks=[
            DocumentBlock(block_id="b1", text="Sample block one", page=1),
            DocumentBlock(block_id="b2", text="Sample block two", page=2),
        ],
    )
    chunks = pipeline.chunk_document(doc)
    assert len(chunks) == 2
    assert chunks[0].document_id == "doc_pipe"
    assert chunks[0].metadata["filename"] == "sample.txt"
    assert chunks[0].metadata["sha256"] == "abc12345"
