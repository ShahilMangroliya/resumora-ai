import pytest
from pydantic import ValidationError

from pipeline.ingestion.models import JobDoc, ResumeDoc


def test_resume_doc_holds_text_format_and_metadata():
    doc = ResumeDoc(
        raw_text="Jane Doe",
        source_format="pdf",
        filename="jane.pdf",
        char_count=8,
        page_count=2,
    )
    assert doc.raw_text == "Jane Doe"
    assert doc.source_format == "pdf"
    assert doc.filename == "jane.pdf"
    assert doc.char_count == 8
    assert doc.page_count == 2


def test_resume_doc_page_count_defaults_to_none():
    doc = ResumeDoc(
        raw_text="Jane Doe",
        source_format="txt",
        filename="jane.txt",
        char_count=8,
    )
    assert doc.page_count is None


def test_resume_doc_rejects_unknown_source_format():
    with pytest.raises(ValidationError):
        ResumeDoc(
            raw_text="Jane Doe",
            source_format="rtf",
            filename="jane.rtf",
            char_count=8,
        )


def test_job_doc_holds_text_and_char_count():
    doc = JobDoc(raw_text="We are hiring", char_count=13)
    assert doc.raw_text == "We are hiring"
    assert doc.char_count == 13
