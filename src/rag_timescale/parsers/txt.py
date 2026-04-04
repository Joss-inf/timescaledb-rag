from __future__ import annotations

from typing import Any
import chardet
from structlog import get_logger
from rag_timescale.parsers.base import BaseParser, ParsedDocument

log = get_logger()

class TextParser(BaseParser):
    def parse(self, file_path: str | bytes, metadata: dict[str, Any] | None = None) -> ParsedDocument:
        try:
            if isinstance(file_path, bytes):
                # Détection automatique de l'encodage
                detected = chardet.detect(file_path)
                encoding = detected.get("encoding", "utf-8") or "utf-8"
                confidence = detected.get("confidence", 0.0)
                
                log.debug("text_encoding_detected", encoding=encoding, confidence=confidence)
                text = file_path.decode(encoding, errors="replace")
            else:
                # Lecture depuis le disque (via le streaming temp file)
                with open(file_path, "rb") as f:
                    content = f.read()
                    detected = chardet.detect(content)
                    encoding = detected.get("encoding", "utf-8") or "utf-8"
                    text = content.decode(encoding, errors="replace")

            return ParsedDocument(
                text=text,
                metadata=metadata or {},
                mime_type="text/plain",
            )
        except Exception as e:
            log.error("text_parsing_failed", error=str(e))
            raise ValueError(f"Erreur lors de la lecture du texte : {str(e)}")

    def supported_extensions(self) -> set[str]:
        return {".txt", ".md", ".text", ".log", ".csv", ".json", ".xml", ".rst"}

    def supported_mime_types(self) -> set[str]:
        return {"text/plain", "text/markdown", "text/csv", "application/json", "text/xml", "application/xml"}
