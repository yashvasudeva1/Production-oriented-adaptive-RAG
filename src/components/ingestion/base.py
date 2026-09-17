from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from .config import IngestionConfig
from .models import Document, DocumentBlock


class BaseLoader(ABC):
    name: str = "base"
    extensions: tuple[str, ...] = ()

    def __init__(self, config: IngestionConfig):
        self.config = config

    @abstractmethod
    def load_blocks(self, path: Path) -> Iterable[DocumentBlock]:
        raise NotImplementedError

    def can_load(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions
