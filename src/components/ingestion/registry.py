from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .base import BaseLoader
from .config import IngestionConfig
from .exceptions import UnsupportedFileTypeError

from .loaders.text import PlainTextLoader, MarkdownLoader
from .loaders.pdf import PDFLoader
from .loaders.office import DOCXLoader, PPTXLoader, ODTLoader
from .loaders.spreadsheet import SpreadsheetLoader, CSVLoader
from .loaders.web import HTMLLoader
from .loaders.data import JSONLoader, XMLLoader, YAMLLoader
from .loaders.image import ImageOCRLoader
from .loaders.email import EMLLoader, MSGLoader
from .loaders.code import CodeLoader
from .loaders.epub import EPUBLoader
from .loaders.media import MediaTranscriptionLoader
from .loaders.unstructured_fallback import UnstructuredFallbackLoader


class LoaderRegistry:
    def __init__(self, config: IngestionConfig):
        self.config = config
        self._loaders: list[BaseLoader] = [
            PDFLoader(config),
            DOCXLoader(config),
            PPTXLoader(config),
            ODTLoader(config),
            SpreadsheetLoader(config),
            CSVLoader(config),
            HTMLLoader(config),
            JSONLoader(config),
            XMLLoader(config),
            YAMLLoader(config),
            ImageOCRLoader(config),
            EMLLoader(config),
            MSGLoader(config),
            EPUBLoader(config),
            MediaTranscriptionLoader(config),
            MarkdownLoader(config),
            CodeLoader(config),
            PlainTextLoader(config),
            # Must be last: specialized loaders should always win.
            UnstructuredFallbackLoader(config),
        ]

    def register(self, loader: BaseLoader) -> None:
        self._loaders.insert(0, loader)

    def get(self, path: Path) -> BaseLoader:
        for loader in self._loaders:
            if loader.can_load(path):
                return loader
        raise UnsupportedFileTypeError(
            f"No ingestion loader registered for: {path.suffix or path.name}"
        )

    def all(self) -> Iterable[BaseLoader]:
        return tuple(self._loaders)
