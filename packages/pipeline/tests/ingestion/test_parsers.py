import pytest

from pipeline.ingestion.errors import IngestionError
from pipeline.ingestion.parsers import extract_docx_text, extract_pdf_text


def test_extract_pdf_text_returns_text_and_page_count(resume_pdf_bytes):
    text, page_count = extract_pdf_text(resume_pdf_bytes)
    assert "Senior Python Engineer" in text
    assert page_count == 1


def test_extract_pdf_text_returns_empty_for_textless_pdf(blank_pdf_bytes):
    text, page_count = extract_pdf_text(blank_pdf_bytes)
    assert text.strip() == ""
    assert page_count == 1


def test_extract_pdf_text_raises_on_corrupt_bytes():
    with pytest.raises(IngestionError, match="PDF"):
        extract_pdf_text(b"this is not a pdf")


def test_extract_docx_text_returns_text(resume_docx_bytes):
    text = extract_docx_text(resume_docx_bytes)
    assert "Senior Python Engineer" in text


def test_extract_docx_text_raises_on_corrupt_bytes():
    with pytest.raises(IngestionError, match="DOCX"):
        extract_docx_text(b"this is not a docx")
