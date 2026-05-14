import io

from docx import Document
from pypdf import PdfReader

from pipeline.ingestion.errors import IngestionError


def extract_pdf_text(data: bytes) -> tuple[str, int]:
    """Extract raw text and page count from PDF bytes.

    Returns un-normalized text — the caller normalizes. A PDF with no
    text layer (e.g. a scan) returns empty text and a valid page count;
    the empty-document check happens during normalization.
    """
    try:
        reader = PdfReader(io.BytesIO(data))
        page_count = len(reader.pages)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        raise IngestionError(f"Could not read PDF file: {exc}") from exc
    return text, page_count


def extract_docx_text(data: bytes) -> str:
    """Extract raw text from DOCX bytes — one paragraph per line.

    Returns un-normalized text — the caller normalizes.
    """
    try:
        doc = Document(io.BytesIO(data))
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)
    except Exception as exc:
        raise IngestionError(f"Could not read DOCX file: {exc}") from exc
