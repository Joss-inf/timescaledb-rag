from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ParsedDocument:
    text: str
    metadata: dict[str, Any]
    mime_type: str


class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: str | bytes, metadata: dict[str, Any] | None = None) -> ParsedDocument:
        pass

    @abstractmethod
    def supported_extensions(self) -> set[str]:
        pass

    @abstractmethod
    def supported_mime_types(self) -> set[str]:
        pass
