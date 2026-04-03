from __future__ import annotations

from typing import Any

import chardet

from rag_timescale.parsers.base import BaseParser, ParsedDocument


class TextParser(BaseParser):
    def parse(self, file_path: str | bytes, metadata: dict[str, Any] | None = None) -> ParsedDocument:
        if isinstance(file_path, bytes):
            detected = chardet.detect(file_path)
            encoding = detected.get("encoding", "utf-8") or "utf-8"
            text = file_path.decode(encoding, errors="replace")
        else:
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

        return ParsedDocument(
            text=text,
            metadata=metadata or {},
            mime_type="text/plain",
        )

    def supported_extensions(self) -> set[str]:
        return {".txt", ".md", ".text", ".log", ".csv", ".json", ".xml", ".rst"}

    def supported_mime_types(self) -> set[str]:
        return {"text/plain", "text/markdown", "text/csv", "application/json", "text/xml", "application/xml"}
