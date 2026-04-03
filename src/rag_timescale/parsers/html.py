from __future__ import annotations

from typing import Any

from bs4 import BeautifulSoup

from rag_timescale.parsers.base import BaseParser, ParsedDocument


class HTMLParser(BaseParser):
    def parse(self, file_path: str | bytes, metadata: dict[str, Any] | None = None) -> ParsedDocument:
        if isinstance(file_path, bytes):
            soup = BeautifulSoup(file_path, "lxml")
        else:
            with open(file_path, encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "lxml")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text_parts: list[str] = []
        for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "pre"]):
            tag_name = element.name
            if tag_name.startswith("h"):
                level = tag_name[1]
                text_parts.append(f"{'#' * int(level)} {element.get_text(strip=True)}")
            elif tag_name == "blockquote":
                text_parts.append(f"> {element.get_text(strip=True)}")
            elif tag_name == "pre":
                text_parts.append(f"```\n{element.get_text(strip=True)}\n```")
            else:
                text_parts.append(element.get_text(strip=True))

        full_text = "\n\n".join(text_parts)

        title = soup.find("title")
        doc_metadata = metadata or {}
        if title:
            doc_metadata["title"] = title.get_text(strip=True)

        return ParsedDocument(
            text=full_text,
            metadata=doc_metadata,
            mime_type="text/html",
        )

    def supported_extensions(self) -> set[str]:
        return {".html", ".htm"}

    def supported_mime_types(self) -> set[str]:
        return {"text/html", "application/xhtml+xml"}
