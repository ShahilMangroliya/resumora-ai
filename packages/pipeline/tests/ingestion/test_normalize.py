import pytest

from pipeline.ingestion.errors import IngestionError
from pipeline.ingestion.normalize import normalize_text


def test_collapses_runs_of_spaces_and_tabs():
    assert normalize_text("Python   \t  Engineer") == "Python Engineer"


def test_collapses_three_or_more_newlines_to_two():
    assert normalize_text("a\n\n\n\n\nb") == "a\n\nb"


def test_trims_each_line_and_overall():
    assert normalize_text("  hello  \n  world  ") == "hello\nworld"


def test_strips_control_characters_but_keeps_newlines_and_tabs():
    # \x00 is a control char and must go; the newline must survive.
    assert normalize_text("a\x00b\nc") == "ab\nc"


def test_applies_nfkc_unicode_normalization():
    # U+FF21 (fullwidth A) normalizes to ASCII "A" under NFKC.
    assert normalize_text("ＡBC") == "ABC"


def test_raises_on_empty_string():
    with pytest.raises(IngestionError, match="no readable text"):
        normalize_text("")


def test_raises_on_whitespace_only_string():
    with pytest.raises(IngestionError, match="no readable text"):
        normalize_text("   \n\t  \n  ")
