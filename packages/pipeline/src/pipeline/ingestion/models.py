from typing import Literal

from pydantic import BaseModel

SourceFormat = Literal["pdf", "docx", "txt"]


class ResumeDoc(BaseModel):
    """A resume after ingestion: normalized text plus lightweight metadata."""

    raw_text: str
    source_format: SourceFormat
    filename: str
    char_count: int
    page_count: int | None = None


class JobDoc(BaseModel):
    """A job description after ingestion: normalized text plus metadata."""

    raw_text: str
    char_count: int
