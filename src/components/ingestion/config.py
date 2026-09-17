from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(slots=True)
class IngestionConfig:
    # Safety / resource limits
    max_file_size_mb: int = 100
    max_archive_files: int = 500
    max_archive_total_size_mb: int = 500

    # Processing
    normalize_unicode: bool = True
    preserve_whitespace_in_code: bool = True
    ocr_images: bool = True
    ocr_pdf_pages: bool = True
    ocr_pdf_min_chars: int = 30
    transcribe_audio_video: bool = False
    whisper_model: str = "base"
    whisper_device: str = "auto"
    whisper_compute_type: str = "default"
    whisper_language: str | None = None
    whisper_word_timestamps: bool = True

    # PDF
    pdf_ocr_dpi: int = 200
    pdf_sort_text: bool = True

    # Spreadsheets
    spreadsheet_max_rows: int = 50_000
    spreadsheet_max_columns: int = 200

    # Archives
    recurse_archives: bool = True

    # Supported archives/media can be extended without changing the pipeline.
    archive_suffixes: Tuple[str, ...] = (
        ".zip", ".tar", ".gz", ".tgz", ".bz2", ".tbz2", ".xz", ".txz"
    )

    tesseract_cmd: str | None = None
    extra_metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        import os
        if self.tesseract_cmd is None:
            env_cmd = os.getenv("TESSERACT_CMD")
            if env_cmd and Path(env_cmd).exists():
                self.tesseract_cmd = env_cmd
            elif Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe").exists():
                self.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    def validate_path(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        if not path.is_file():
            raise ValueError(f"Expected a file, got: {path}")

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            raise ValueError(
                f"{path.name} is {size_mb:.1f} MB; "
                f"limit is {self.max_file_size_mb} MB"
            )