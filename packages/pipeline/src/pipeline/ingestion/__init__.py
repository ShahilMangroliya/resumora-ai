from pathlib import Path

from pipeline.ingestion.errors import IngestionError
from pipeline.ingestion.models import JobDoc, ResumeDoc
from pipeline.ingestion.normalize import normalize_text
from pipeline.ingestion.parsers import extract_docx_text, extract_pdf_text

__all__ = ["IngestionError", "JobDoc", "ResumeDoc", "ingest_resume"]


def ingest_resume(data: bytes, filename: str) -> ResumeDoc:
    """Ingest a resume file into a normalized ResumeDoc.

    The format is detected from the filename extension (.pdf, .docx,
    .txt). Raises IngestionError for an unsupported extension, an
    unreadable file, or a document with no readable text.
    """
    ext = Path(filename).suffix.lower().lstrip(".")
    page_count: int | None = None

    if ext == "pdf":
        raw_text, page_count = extract_pdf_text(data)
    elif ext == "docx":
        raw_text = extract_docx_text(data)
    elif ext == "txt":
        raw_text = data.decode("utf-8", errors="replace")
    else:
        raise IngestionError(f"Unsupported file format: '{filename}'")

    text = normalize_text(raw_text)
    return ResumeDoc(
        raw_text=text,
        source_format=ext,
        filename=filename,
        char_count=len(text),
        page_count=page_count,
    )
