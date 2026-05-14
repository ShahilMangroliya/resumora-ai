import pytest

from pipeline.ingestion import IngestionError, JobDoc, ingest_job


def test_ingest_job_returns_job_doc():
    doc = ingest_job("We are hiring a Senior Python Engineer.")
    assert isinstance(doc, JobDoc)
    assert "Senior Python Engineer" in doc.raw_text
    assert doc.char_count == len(doc.raw_text)


def test_ingest_job_normalizes_whitespace():
    doc = ingest_job("Python    Engineer\n\n\n\nRemote")
    assert doc.raw_text == "Python Engineer\n\nRemote"


def test_ingest_job_rejects_empty_text():
    with pytest.raises(IngestionError, match="no readable text"):
        ingest_job("   \n\t  ")
