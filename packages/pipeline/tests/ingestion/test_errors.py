import pytest

from pipeline.ingestion.errors import IngestionError


def test_ingestion_error_is_an_exception():
    assert issubclass(IngestionError, Exception)


def test_ingestion_error_carries_a_message():
    with pytest.raises(IngestionError, match="bad file"):
        raise IngestionError("bad file")
