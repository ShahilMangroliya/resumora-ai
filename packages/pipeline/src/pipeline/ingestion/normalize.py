import re
import unicodedata

from pipeline.ingestion.errors import IngestionError

_KEEP_CONTROL = {"\n", "\t"}


def normalize_text(raw: str) -> str:
    """Normalize extracted text to clean, readable plain text.

    Applies NFKC unicode normalization, removes control characters
    (except newline and tab), collapses runs of spaces/tabs and blank
    lines, and trims whitespace. Raises IngestionError if no readable
    text remains.
    """
    text = unicodedata.normalize("NFKC", raw)
    text = "".join(
        ch
        for ch in text
        if ch in _KEEP_CONTROL or unicodedata.category(ch)[0] != "C"
    )
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = text.strip()
    if not text:
        raise IngestionError("Document contains no readable text")
    return text
