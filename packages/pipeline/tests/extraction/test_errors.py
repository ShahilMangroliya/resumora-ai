import pytest

from pipeline.extraction.errors import ExtractionError


def test_extraction_error_is_an_exception():
    assert issubclass(ExtractionError, Exception)


def test_extraction_error_carries_a_message():
    with pytest.raises(ExtractionError, match="ollama down"):
        raise ExtractionError("ollama down")
