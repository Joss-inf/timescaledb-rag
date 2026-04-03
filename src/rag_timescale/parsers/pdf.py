from __future__ import annotations

from typing import Any

import pdfplumber

from rag_timescale.parsers.base import BaseParser, ParsedDocument


class PDFParser(BaseParser):
    def parse(self, file_path: str | bytes, metadata: dict[str, Any] | None = None) -> ParsedDocument:
        text_parts: list[str] = []
        page_texts: list[str] = []

        if isinstance(file_path, bytes):
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    page_texts.append(page_text)
                    text_parts.append(page_text)
        else:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    page_texts.append(page_text)
                    text_parts.append(page_text)

        full_text = "\n\n".join(text_parts)
        doc_metadata = metadata or {}
        doc_metadata["page_count"] = len(page_texts)
        doc_metadata["page_texts"] = page_texts

        return ParsedDocument(
            text=full_text,
            metadata=doc_metadata,
            mime_type="application/pdf",
        )

    def supported_extensions(self) -> set[str]:
        return {".pdf"}

    def supported_mime_types(self) -> set[str]:
        return {"application/pdf"}
