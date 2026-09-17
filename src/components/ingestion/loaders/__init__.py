from .text import PlainTextLoader, MarkdownLoader
from .pdf import PDFLoader
from .office import DOCXLoader, PPTXLoader, ODTLoader
from .spreadsheet import SpreadsheetLoader, CSVLoader
from .web import HTMLLoader
from .data import JSONLoader, XMLLoader, YAMLLoader
from .image import ImageOCRLoader
from .email import EMLLoader, MSGLoader
from .code import CodeLoader
from .epub import EPUBLoader

__all__ = [
    "PlainTextLoader",
    "MarkdownLoader",
    "PDFLoader",
    "DOCXLoader",
    "PPTXLoader",
    "ODTLoader",
    "SpreadsheetLoader",
    "CSVLoader",
    "HTMLLoader",
    "JSONLoader",
    "XMLLoader",
    "YAMLLoader",
    "ImageOCRLoader",
    "EMLLoader",
    "MSGLoader",
    "CodeLoader",
    "EPUBLoader",
]

from .media import MediaTranscriptionLoader
from .unstructured_fallback import UnstructuredFallbackLoader
