# Modular RAG Ingestion Layer

Location:

```text
src/
└── ingestion/
    ├── config.py
    ├── exceptions.py
    ├── models.py
    ├── base.py
    ├── registry.py
    ├── pipeline.py
    ├── utils.py
    └── loaders/
        ├── text.py
        ├── pdf.py
        ├── office.py
        ├── spreadsheet.py
        ├── web.py
        ├── data.py
        ├── image.py
        ├── email.py
        ├── code.py
        └── epub.py
```

## Supported formats

- PDF: `.pdf`
- Word: `.docx`
- PowerPoint: `.pptx`
- OpenDocument text: `.odt`
- Excel: `.xlsx`, `.xlsm`, `.xltx`, `.xltm`
- CSV / TSV: `.csv`, `.tsv`
- HTML/XHTML: `.html`, `.htm`, `.xhtml`
- JSON / JSONL / NDJSON
- XML / RSS / Atom
- YAML
- Markdown
- Plain text / logs
- Source code and config files
- Images with OCR: PNG/JPEG/WEBP/BMP/TIFF/GIF
- EML
- MSG
- EPUB
- Audio/video transcription: MP3/WAV/M4A/FLAC/OGG/MP4/MKV/MOV/WEBM and common variants (optional faster-whisper)
- Unstructured fallback for additional file types when no dedicated loader matches
- ZIP/TAR-style archives (recursive ingestion)

## Important design decision

This layer does **not** make retrieval chunks.

It emits structured `DocumentBlock` objects while retaining:

- page number
- slide number
- spreadsheet sheet
- section/heading
- source locator
- block type
- loader
- document SHA-256

Your chunker can then combine blocks while retaining these provenance fields.

## Minimal usage

```python
from src.ingestion import IngestionPipeline

pipeline = IngestionPipeline()

doc = pipeline.ingest_file("data/report.pdf")

print(doc.filename)
print(doc.word_count)

for block in doc.blocks:
    print(block.block_type, block.source_locator)
    print(block.text[:500])
```

## Batch ingestion

```python
from pathlib import Path
from src.ingestion import IngestionPipeline

pipeline = IngestionPipeline()

paths = list(Path("data").rglob("*"))
files = [p for p in paths if p.is_file()]

result = pipeline.ingest_many(files)

print("Documents:", len(result.documents))
print("Skipped:", result.skipped)
print("Errors:", result.errors)
```

## OCR

For scanned PDFs, the PDF loader attempts PyMuPDF OCR when extracted text is too short. PyMuPDF supports normal text extraction and page OCR through `get_textpage_ocr()`.

For image files, `pytesseract` is used with Pillow.

You still need the **Tesseract executable** installed on the machine.

## Next layer

Recommended next architecture:

```text
INGESTION
    ↓
STRUCTURE-AWARE CHUNKER
    ↓
METADATA ENRICHMENT
    ↓
EMBEDDINGS
    ↓
QDRANT
```

The key is to chunk after ingestion, not inside every file parser.


## Broad fallback

`UnstructuredFallbackLoader` is the final loader in the registry. It catches file types without a dedicated parser and converts Unstructured elements into the same `DocumentBlock` schema. Unstructured currently documents support for 65+ file types.

## Media transcription

```python
from src.ingestion import IngestionConfig, IngestionPipeline

config = IngestionConfig(
    transcribe_audio_video=True,
    whisper_model="base",
    whisper_device="auto",
    whisper_compute_type="default",
)

pipeline = IngestionPipeline(config)
doc = pipeline.ingest_file("recording.mp3")
```
