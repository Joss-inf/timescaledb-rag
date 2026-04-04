from __future__ import annotations

from typing import Any
import pdfplumber
from rag_timescale.parsers.base import BaseParser, ParsedDocument
import io


class PDFParser(BaseParser):
    def parse(self, file_path: str | bytes, metadata: dict[str, Any] | None = None) -> ParsedDocument:
        text_parts: list[str] = []
        doc_metadata = metadata or {}
        
        try:
            source = io.BytesIO(file_path) if isinstance(file_path, bytes) else file_path
            
            with pdfplumber.open(source) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text_parts.append(f"[[PAGE_{i+1}]]\n{page_text}")
                
                doc_metadata["page_count"] = len(pdf.pages)
                
        except Exception as e:
            raise ValueError(f"Erreur lors de l'extraction du PDF : {str(e)}")

        full_text = "\n\n".join(text_parts)

        return ParsedDocument(
            text=full_text,
            metadata=doc_metadata,
            mime_type="application/pdf",
        )

    def supported_extensions(self) -> set[str]:
        return {".pdf"}

    def supported_mime_types(self) -> set[str]:
        return {"application/pdf"}
