import pytest

from pipeline.ingestion import IngestionError, ResumeDoc, ingest_resume


def test_ingest_resume_from_pdf(resume_pdf_bytes):
    doc = ingest_resume(resume_pdf_bytes, "jane.pdf")
    assert isinstance(doc, ResumeDoc)
    assert doc.source_format == "pdf"
    assert doc.filename == "jane.pdf"
    assert "Senior Python Engineer" in doc.raw_text
    assert doc.char_count == len(doc.raw_text)
    assert doc.page_count == 1


def test_ingest_resume_from_docx(resume_docx_bytes):
    doc = ingest_resume(resume_docx_bytes, "jane.docx")
    assert doc.source_format == "docx"
    assert "Senior Python Engineer" in doc.raw_text
    assert doc.page_count is None


def test_ingest_resume_from_txt(resume_txt_bytes):
    doc = ingest_resume(resume_txt_bytes, "jane.txt")
    assert doc.source_format == "txt"
    assert "Senior Python Engineer" in doc.raw_text
    assert doc.page_count is None


def test_ingest_resume_detects_extension_case_insensitively(resume_pdf_bytes):
    doc = ingest_resume(resume_pdf_bytes, "JANE.PDF")
    assert doc.source_format == "pdf"


def test_ingest_resume_rejects_unsupported_extension(resume_txt_bytes):
    with pytest.raises(IngestionError, match="Unsupported"):
        ingest_resume(resume_txt_bytes, "jane.rtf")


def test_ingest_resume_rejects_missing_extension(resume_txt_bytes):
    with pytest.raises(IngestionError, match="no extension"):
        ingest_resume(resume_txt_bytes, "resume_no_ext")


def test_ingest_resume_rejects_scanned_pdf(blank_pdf_bytes):
    with pytest.raises(IngestionError, match="no readable text"):
        ingest_resume(blank_pdf_bytes, "scan.pdf")


def test_ingest_resume_rejects_corrupt_pdf():
    with pytest.raises(IngestionError, match="Could not read PDF"):
        ingest_resume(b"this is not a pdf", "jane.pdf")
