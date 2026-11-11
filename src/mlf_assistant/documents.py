from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from uuid import UUID

from mlf_assistant import db
from mlf_assistant.config import get_settings
from mlf_assistant.embeddings import embed_texts
from mlf_assistant.schemas import DocumentMetadata


SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".xlsx", ".csv", ".pptx", ".txt", ".md", ".html",
    ".htm", ".xml", ".json", ".png", ".jpg", ".jpeg", ".tif", ".tiff",
}


class DocumentProcessingError(ValueError):
    pass


def ingest_document(
    filename: str, content: bytes, metadata: DocumentMetadata, *, actor: str
) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise DocumentProcessingError("This file type is not supported.")
    if not content or len(content) > get_settings().max_upload_mb * 1024 * 1024:
        raise DocumentProcessingError("The document is empty or exceeds the upload size limit.")
    clean_name = Path(filename).name
    digest = hashlib.sha256(content).hexdigest()
    if db.document_exists(digest):
        document_id = db.retry_failed_document(
            metadata.model_dump(), digest, clean_name, actor
        )
        if not document_id:
            raise DocumentProcessingError("This document has already been uploaded.")
    else:
        document_id = db.store_document(metadata.model_dump(), digest, clean_name, actor)
    try:
        text = _extract_text(suffix, content)
        chunks = _chunk_document(text)
        if not chunks:
            raise DocumentProcessingError("No readable text was extracted from this document.")
        vectors = embed_texts(chunks)
        db.save_chunks(document_id, chunks, vectors)
    except Exception as exc:
        try:
            db.set_ingestion_status(document_id, "failed", str(exc))
        except Exception:
            pass
        if isinstance(exc, DocumentProcessingError):
            raise
        raise DocumentProcessingError("Document processing failed; the document remains unavailable.") from exc
    return document_id


def reindex_document(document_id: str) -> int:
    try:
        document_id = str(UUID(document_id))
    except ValueError as exc:
        raise DocumentProcessingError(
            "Select a document by its document ID. A version number is not a document ID."
        ) from exc
    chunks = db.get_document_chunks(document_id)
    if not chunks:
        raise DocumentProcessingError("No extracted chunks are available to re-index.")
    vectors = embed_texts([item["text"] for item in chunks])
    for item, vector in zip(chunks, vectors, strict=True):
        db.update_chunk_embedding(item["id"], vector)
    return len(chunks)


def _extract_text(suffix: str, content: bytes) -> str:
    if suffix in {".txt", ".md"}:
        return content.decode("utf-8-sig", errors="replace")
    if suffix == ".csv":
        rows = csv.reader(io.StringIO(content.decode("utf-8-sig", errors="replace")))
        return "\n".join(" | ".join(cell.strip() for cell in row) for row in rows)
    if suffix == ".json":
        parsed = json.loads(content)
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    if suffix in {".html", ".htm", ".xml"}:
        parser = _TextExtractor()
        parser.feed(content.decode("utf-8-sig", errors="replace"))
        return html.unescape(" ".join(parser.parts))
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            pass
        else:
            reader = PdfReader(io.BytesIO(content))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            if text.strip():
                return text

    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise DocumentProcessingError(
            "Docling is required for PDF, Office, and image uploads. Install the documents extra."
        ) from exc

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=suffix) as temporary:
        temporary.write(content)
        temporary.flush()
        converted = DocumentConverter().convert(temporary.name)
        document = converted.document
        if hasattr(document, "export_to_markdown"):
            return document.export_to_markdown()
        return document.export_to_text()


def _chunk_document(text: str, *, max_chars: int = 1800) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [re.sub(r"\s+", " ", block).strip() for block in re.split(r"\n\s*\n", normalized)]
    blocks = [block for block in blocks if block]
    chunks: list[str] = []
    current = ""
    for block in blocks:
        if len(block) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(block[index : index + max_chars] for index in range(0, len(block), max_chars))
            continue
        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) > max_chars:
            chunks.append(current)
            current = block
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())