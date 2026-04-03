from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from rag_timescale.parsers.base import BaseParser, ParsedDocument
from rag_timescale.parsers.docx import DOCXParser
from rag_timescale.parsers.html import HTMLParser
from rag_timescale.parsers.pdf import PDFParser
from rag_timescale.parsers.txt import TextParser

_REGISTRY: dict[str, BaseParser] = {
    "application/pdf": PDFParser(),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DOCXParser(),
    "text/html": HTMLParser(),
    "text/plain": TextParser(),
    "text/markdown": TextParser(),
}

_EXTENSION_MAP = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".html": "text/html",
    ".htm": "text/html",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".text": "text/plain",
    ".log": "text/plain",
    ".csv": "text/plain",
    ".json": "text/plain",
    ".xml": "text/plain",
    ".rst": "text/plain",
}


def get_parser(file_path: str | None = None, mime_type: str | None = None) -> BaseParser:
    if mime_type and mime_type in _REGISTRY:
        return _REGISTRY[mime_type]

    if file_path:
        ext = Path(file_path).suffix.lower()
        if ext in _EXTENSION_MAP:
            guessed_mime = _EXTENSION_MAP[ext]
            if guessed_mime in _REGISTRY:
                return _REGISTRY[guessed_mime]

        guessed_mime, _ = mimetypes.guess_type(file_path)
        if guessed_mime and guessed_mime in _REGISTRY:
            return _REGISTRY[guessed_mime]

    return _REGISTRY["text/plain"]


def parse_document(
    file_path: str | bytes,
    filename: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ParsedDocument:
    parser = get_parser(file_path=filename if isinstance(file_path, bytes) else file_path)
    return parser.parse(file_path, metadata)


def register_parser(mime_type: str, parser: BaseParser) -> None:
    _REGISTRY[mime_type] = parser


def supported_mime_types() -> list[str]:
    return list(_REGISTRY.keys())


def supported_extensions() -> list[str]:
    return list(_EXTENSION_MAP.keys())
