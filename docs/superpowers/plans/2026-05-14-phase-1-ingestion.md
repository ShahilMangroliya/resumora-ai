# Phase 1 — Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `pipeline/ingestion` — turn a resume file (PDF/DOCX/text) and a pasted job description into normalized, validated `ResumeDoc` / `JobDoc` objects.

**Architecture:** A pure Python sub-package inside `packages/pipeline` with no web/server knowledge. Two public functions: `ingest_resume(data: bytes, filename: str)` and `ingest_job(text: str)`. Format is detected from the filename extension. Every text path runs through one normalization function. All failure modes raise a single `IngestionError` that the API (Phase 6) will map to HTTP 400. "Clean text only" — no section detection, no layout, no OCR.

**Tech Stack:** Python 3.12, `pypdf` (PDF text extraction), `python-docx` (DOCX text extraction), `pydantic` v2 (data models), `pytest`, `reportlab` (test-fixture generation only — dev dependency).

**This plan is Phase 1 only.** It follows the master design doc `docs/superpowers/specs/2026-05-14-ai-pipeline-design.md` (§4 component 1, §7 phase 1). Decisions locked in during brainstorming on 2026-05-14:

- **Clean text only** — extract and normalize text + lightweight metadata. No section splitting (that is Phase 2's Ollama extraction).
- **JD is text-only** — `ingest_job` takes a `str`; only the resume path parses files.
- **Input interface is bytes + filename** — matches how FastAPI hands off an upload; tests pass raw bytes with no temp files.
- **Scanned / image-only PDFs raise `IngestionError`** — `pypdf` yields empty text, which is treated as an empty document. No OCR.
- **Pydantic models** for `ResumeDoc` / `JobDoc` — consistent with FastAPI downstream, free validation, still framework-agnostic as pure data.

---

## File Structure

Files created or modified in this phase and their responsibility:

- `packages/pipeline/pyproject.toml` — **modify**: add `pypdf`, `python-docx`, `pydantic` runtime dependencies.
- `pyproject.toml` (workspace root) — **modify**: add `reportlab` to the `dev` dependency group (test fixtures only).
- `packages/pipeline/src/pipeline/ingestion/__init__.py` — **create**: the ingestion sub-package; public API (`ingest_resume`, `ingest_job`) and re-exports of the models and `IngestionError`.
- `packages/pipeline/src/pipeline/ingestion/errors.py` — **create**: the `IngestionError` exception.
- `packages/pipeline/src/pipeline/ingestion/normalize.py` — **create**: `normalize_text` — unicode/whitespace cleanup; raises `IngestionError` on empty result.
- `packages/pipeline/src/pipeline/ingestion/models.py` — **create**: `ResumeDoc` and `JobDoc` Pydantic models.
- `packages/pipeline/src/pipeline/ingestion/parsers.py` — **create**: `extract_pdf_text` and `extract_docx_text` — format-specific byte→text extraction.
- `packages/pipeline/tests/conftest.py` — **create**: session-scoped pytest fixtures that generate sample resume bytes (PDF with text, blank PDF, DOCX, txt).
- `packages/pipeline/tests/ingestion/test_errors.py` — **create**: tests for `IngestionError`.
- `packages/pipeline/tests/ingestion/test_normalize.py` — **create**: tests for `normalize_text`.
- `packages/pipeline/tests/ingestion/test_models.py` — **create**: tests for the data models.
- `packages/pipeline/tests/ingestion/test_parsers.py` — **create**: tests for `extract_pdf_text` / `extract_docx_text`.
- `packages/pipeline/tests/ingestion/test_ingest_resume.py` — **create**: tests for the `ingest_resume` public function.
- `packages/pipeline/tests/ingestion/test_ingest_job.py` — **create**: tests for the `ingest_job` public function.

---

## Task 1: Dependencies and ingestion package skeleton

**Files:**
- Modify: `packages/pipeline/pyproject.toml`
- Modify: `pyproject.toml` (workspace root)
- Create: `packages/pipeline/src/pipeline/ingestion/__init__.py`

- [ ] **Step 1: Add runtime dependencies to the pipeline package**

In `packages/pipeline/pyproject.toml`, change:

```toml
dependencies = []
```

to:

```toml
dependencies = [
    "pypdf>=5.0",
    "python-docx>=1.1",
    "pydantic>=2.0",
]
```

- [ ] **Step 2: Add `reportlab` to the root dev dependency group**

In the workspace root `pyproject.toml`, change the `[dependency-groups]` block from:

```toml
[dependency-groups]
dev = [
    "pytest>=8.0",
    "httpx>=0.27",
    "ruff>=0.6",
    "honcho>=2.0",
]
```

to:

```toml
[dependency-groups]
dev = [
    "pytest>=8.0",
    "httpx>=0.27",
    "ruff>=0.6",
    "honcho>=2.0",
    "reportlab>=4.0",
]
```

`reportlab` is used only to generate PDF test fixtures (Task 5) — it is not a runtime dependency of the pipeline.

- [ ] **Step 3: Create the ingestion sub-package marker**

Create `packages/pipeline/src/pipeline/ingestion/__init__.py` as an empty file (the public API is filled in by Task 8 and Task 9).

- [ ] **Step 4: Sync the workspace**

Run: `uv sync --all-packages`
Expected: completes without error; `pypdf`, `python-docx`, `pydantic`, and `reportlab` are installed; `uv.lock` is updated.

- [ ] **Step 5: Verify the new package imports**

Run: `uv run python -c "import pipeline.ingestion; import pypdf; import docx; import pydantic; import reportlab; print('ok')"`
Expected: prints `ok` with no `ModuleNotFoundError`.

- [ ] **Step 6: Commit**

```bash
git add packages/pipeline/pyproject.toml pyproject.toml packages/pipeline/src/pipeline/ingestion/__init__.py uv.lock
git commit -m "feat: add ingestion dependencies and package skeleton"
```

---

## Task 2: IngestionError exception

A single exception type for every ingestion failure mode. The API layer (Phase 6) catches this one type and maps it to HTTP 400.

**Files:**
- Create: `packages/pipeline/src/pipeline/ingestion/errors.py`
- Test: `packages/pipeline/tests/ingestion/test_errors.py`

- [ ] **Step 1: Write the failing test**

Create `packages/pipeline/tests/ingestion/test_errors.py`:

```python
import pytest

from pipeline.ingestion.errors import IngestionError


def test_ingestion_error_is_an_exception():
    assert issubclass(IngestionError, Exception)


def test_ingestion_error_carries_a_message():
    with pytest.raises(IngestionError, match="bad file"):
        raise IngestionError("bad file")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/pipeline/tests/ingestion/test_errors.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.ingestion.errors'`.

- [ ] **Step 3: Write the exception**

Create `packages/pipeline/src/pipeline/ingestion/errors.py`:

```python
class IngestionError(Exception):
    """Raised when a resume or job description cannot be ingested.

    The API layer maps this to an HTTP 400 response.
    """
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest packages/pipeline/tests/ingestion/test_errors.py -v`
Expected: PASS — 2 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/pipeline/src/pipeline/ingestion/errors.py packages/pipeline/tests/ingestion/test_errors.py
git commit -m "feat: add IngestionError exception"
```

---

## Task 3: Text normalization

`normalize_text` is the single chokepoint every text path runs through. It does unicode normalization (NFKC), strips control characters, collapses runs of whitespace, trims each line, and — critically — raises `IngestionError` if nothing readable is left. The "scanned PDF" and "empty input" failure modes are both enforced here, so callers never have to re-check.

**Files:**
- Create: `packages/pipeline/src/pipeline/ingestion/normalize.py`
- Test: `packages/pipeline/tests/ingestion/test_normalize.py`

- [ ] **Step 1: Write the failing test**

Create `packages/pipeline/tests/ingestion/test_normalize.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/pipeline/tests/ingestion/test_normalize.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.ingestion.normalize'`.

- [ ] **Step 3: Write the implementation**

Create `packages/pipeline/src/pipeline/ingestion/normalize.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest packages/pipeline/tests/ingestion/test_normalize.py -v`
Expected: PASS — 7 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/pipeline/src/pipeline/ingestion/normalize.py packages/pipeline/tests/ingestion/test_normalize.py
git commit -m "feat: add text normalization for ingestion"
```

---

## Task 4: Data models

`ResumeDoc` and `JobDoc` are the normalized outputs of ingestion — pure Pydantic data models with no behavior.

**Files:**
- Create: `packages/pipeline/src/pipeline/ingestion/models.py`
- Test: `packages/pipeline/tests/ingestion/test_models.py`

- [ ] **Step 1: Write the failing test**

Create `packages/pipeline/tests/ingestion/test_models.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/pipeline/tests/ingestion/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.ingestion.models'`.

- [ ] **Step 3: Write the implementation**

Create `packages/pipeline/src/pipeline/ingestion/models.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest packages/pipeline/tests/ingestion/test_models.py -v`
Expected: PASS — 4 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/pipeline/src/pipeline/ingestion/models.py packages/pipeline/tests/ingestion/test_models.py
git commit -m "feat: add ResumeDoc and JobDoc models"
```

---

## Task 5: Test fixtures

Session-scoped pytest fixtures that generate sample resume bytes in memory — a PDF with real text, a blank (text-less) PDF standing in for a scanned document, a DOCX, and a plain-text file. No binary files are committed to the repo; everything is generated at test time. Tasks 6–9 consume these fixtures.

**Files:**
- Create: `packages/pipeline/tests/conftest.py`

- [ ] **Step 1: Write the conftest**

Create `packages/pipeline/tests/conftest.py`:

```python
import io

import pytest
from docx import Document
from reportlab.pdfgen import canvas

_RESUME_LINES = [
    "Jane Doe",
    "Senior Python Engineer",
    "Skills: Python, FastAPI, Docker",
]


@pytest.fixture(scope="session")
def resume_pdf_bytes() -> bytes:
    """A PDF resume with an extractable text layer."""
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf)
    y = 720
    for line in _RESUME_LINES:
        pdf.drawString(72, y, line)
        y -= 20
    pdf.showPage()
    pdf.save()
    return buf.getvalue()


@pytest.fixture(scope="session")
def blank_pdf_bytes() -> bytes:
    """A one-page PDF with no text layer — stands in for a scanned resume."""
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf)
    pdf.showPage()
    pdf.save()
    return buf.getvalue()


@pytest.fixture(scope="session")
def resume_docx_bytes() -> bytes:
    """A DOCX resume with one paragraph per line."""
    buf = io.BytesIO()
    doc = Document()
    for line in _RESUME_LINES:
        doc.add_paragraph(line)
    doc.save(buf)
    return buf.getvalue()


@pytest.fixture(scope="session")
def resume_txt_bytes() -> bytes:
    """A plain-text resume."""
    return ("\n".join(_RESUME_LINES) + "\n").encode("utf-8")
```

- [ ] **Step 2: Verify the conftest imports cleanly**

Run: `uv run pytest packages/pipeline/tests/ -v`
Expected: PASS — the existing tests still pass (`test_smoke.py`, `test_errors.py`, `test_normalize.py`, `test_models.py`) and there is no collection error from `conftest.py`. The new fixtures are unused so far, which is fine.

- [ ] **Step 3: Commit**

```bash
git add packages/pipeline/tests/conftest.py
git commit -m "test: add resume sample fixtures for ingestion"
```

---

## Task 6: PDF and DOCX parsers

`extract_pdf_text` and `extract_docx_text` turn raw file bytes into raw (un-normalized) text. They live in one file because they are the same kind of unit — format-specific byte→text extraction — and change together. Each wraps any underlying parsing failure in `IngestionError`. `extract_pdf_text` also returns the page count.

**Files:**
- Create: `packages/pipeline/src/pipeline/ingestion/parsers.py`
- Test: `packages/pipeline/tests/ingestion/test_parsers.py`

- [ ] **Step 1: Write the failing test**

Create `packages/pipeline/tests/ingestion/test_parsers.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/pipeline/tests/ingestion/test_parsers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.ingestion.parsers'`.

- [ ] **Step 3: Write the implementation**

Create `packages/pipeline/src/pipeline/ingestion/parsers.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest packages/pipeline/tests/ingestion/test_parsers.py -v`
Expected: PASS — 5 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/pipeline/src/pipeline/ingestion/parsers.py packages/pipeline/tests/ingestion/test_parsers.py
git commit -m "feat: add PDF and DOCX text parsers"
```

---

## Task 7: `ingest_resume` public function

The first public entry point. Detects the format from the filename extension, dispatches to the right parser (or decodes text directly), normalizes, and returns a `ResumeDoc`. An unsupported extension raises `IngestionError`; a scanned/empty PDF raises `IngestionError` via `normalize_text`.

**Files:**
- Modify: `packages/pipeline/src/pipeline/ingestion/__init__.py`
- Test: `packages/pipeline/tests/ingestion/test_ingest_resume.py`

- [ ] **Step 1: Write the failing test**

Create `packages/pipeline/tests/ingestion/test_ingest_resume.py`:

```python
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


def test_ingest_resume_rejects_scanned_pdf(blank_pdf_bytes):
    with pytest.raises(IngestionError, match="no readable text"):
        ingest_resume(blank_pdf_bytes, "scan.pdf")


def test_ingest_resume_rejects_corrupt_pdf():
    with pytest.raises(IngestionError, match="Could not read PDF"):
        ingest_resume(b"this is not a pdf", "jane.pdf")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/pipeline/tests/ingestion/test_ingest_resume.py -v`
Expected: FAIL — `ImportError: cannot import name 'ingest_resume' from 'pipeline.ingestion'`.

- [ ] **Step 3: Write the implementation**

Replace the contents of `packages/pipeline/src/pipeline/ingestion/__init__.py` with:

```python
from pathlib import Path

from pipeline.ingestion.errors import IngestionError
from pipeline.ingestion.models import JobDoc, ResumeDoc
from pipeline.ingestion.normalize import normalize_text
from pipeline.ingestion.parsers import extract_docx_text, extract_pdf_text

__all__ = ["IngestionError", "JobDoc", "ResumeDoc", "ingest_resume"]


def ingest_resume(data: bytes, filename: str) -> ResumeDoc:
    """Ingest a resume file into a normalized ResumeDoc.

    The format is detected from the filename extension (.pdf, .docx,
    .txt). Raises IngestionError for an unsupported extension, an
    unreadable file, or a document with no readable text.
    """
    ext = Path(filename).suffix.lower().lstrip(".")
    page_count: int | None = None

    if ext == "pdf":
        raw_text, page_count = extract_pdf_text(data)
    elif ext == "docx":
        raw_text = extract_docx_text(data)
    elif ext == "txt":
        raw_text = data.decode("utf-8", errors="replace")
    else:
        raise IngestionError(f"Unsupported file format: '{filename}'")

    text = normalize_text(raw_text)
    return ResumeDoc(
        raw_text=text,
        source_format=ext,
        filename=filename,
        char_count=len(text),
        page_count=page_count,
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest packages/pipeline/tests/ingestion/test_ingest_resume.py -v`
Expected: PASS — 7 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/pipeline/src/pipeline/ingestion/__init__.py packages/pipeline/tests/ingestion/test_ingest_resume.py
git commit -m "feat: add ingest_resume public function"
```

---

## Task 8: `ingest_job` public function

The second public entry point. The job description arrives as already-extracted text (the web UI pastes it), so this just normalizes and wraps it in a `JobDoc`.

**Files:**
- Modify: `packages/pipeline/src/pipeline/ingestion/__init__.py`
- Test: `packages/pipeline/tests/ingestion/test_ingest_job.py`

- [ ] **Step 1: Write the failing test**

Create `packages/pipeline/tests/ingestion/test_ingest_job.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/pipeline/tests/ingestion/test_ingest_job.py -v`
Expected: FAIL — `ImportError: cannot import name 'ingest_job' from 'pipeline.ingestion'`.

- [ ] **Step 3: Write the implementation**

In `packages/pipeline/src/pipeline/ingestion/__init__.py`, update the `__all__` list and add the `ingest_job` function.

Change:

```python
__all__ = ["IngestionError", "JobDoc", "ResumeDoc", "ingest_resume"]
```

to:

```python
__all__ = ["IngestionError", "JobDoc", "ResumeDoc", "ingest_job", "ingest_resume"]
```

Then add this function at the end of the file:

```python
def ingest_job(text: str) -> JobDoc:
    """Ingest a job description (pasted plain text) into a normalized JobDoc.

    Raises IngestionError if the text has no readable content.
    """
    normalized = normalize_text(text)
    return JobDoc(raw_text=normalized, char_count=len(normalized))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest packages/pipeline/tests/ingestion/test_ingest_job.py -v`
Expected: PASS — 3 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/pipeline/src/pipeline/ingestion/__init__.py packages/pipeline/tests/ingestion/test_ingest_job.py
git commit -m "feat: add ingest_job public function"
```

---

## Task 9: Full-suite verification

Confirm the whole phase hangs together — every test, the linter, and the package import surface.

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `make test`
Expected: all tests pass — the Phase 0 tests (`test_smoke.py`, `test_health_returns_ok`) plus every Phase 1 test (`test_errors.py`, `test_normalize.py`, `test_models.py`, `test_parsers.py`, `test_ingest_resume.py`, `test_ingest_job.py`). No failures, no collection errors.

- [ ] **Step 2: Run the linter**

Run: `make lint`
Expected: `ruff check` reports `All checks passed!`.

- [ ] **Step 3: Verify the public API surface**

Run: `uv run python -c "from pipeline.ingestion import ingest_resume, ingest_job, ResumeDoc, JobDoc, IngestionError; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Commit (only if Steps 1–2 required fixes)**

If `make lint` or `make test` surfaced anything that needed a fix, commit it:

```bash
git add -A
git commit -m "fix: address lint and test issues in ingestion"
```

If nothing needed fixing, skip this step — there is nothing to commit.

---

## Self-Review

**1. Spec coverage** (master design doc §4 component 1 + §7 phase 1 — "`pipeline/ingestion`: PDF/DOCX/text → normalized docs + tests"; deps `pypdf`, `python-docx`):
- `pipeline/ingestion` sub-package → Task 1. ✓
- PDF → text → Task 6 (`extract_pdf_text`). ✓
- DOCX → text → Task 6 (`extract_docx_text`). ✓
- text (txt) → Task 7 (`ingest_resume`, `.txt` branch). ✓
- Normalized `ResumeDoc` / `JobDoc` → Task 4 (models) + Task 3 (normalization). ✓
- `pypdf` + `python-docx` dependencies → Task 1. ✓
- Tests for every unit → Tasks 2–8 (TDD) + Task 9 (full-suite). ✓
- Brainstorming decisions: clean text only (no section split — confirmed nowhere in the plan does parsing detect sections ✓); JD text-only (Task 8 takes `str` ✓); bytes + filename interface (Task 7 signature ✓); scanned PDF raises (Task 7 `test_ingest_resume_rejects_scanned_pdf` ✓); Pydantic models (Task 4 ✓).

**2. Placeholder scan:** No "TBD"/"TODO"/"handle edge cases"/"similar to Task N". Every code step shows complete file contents or an exact before/after edit. ✓

**3. Type consistency:**
- `IngestionError` — defined in Task 2 (`errors.py`), imported and raised consistently in Tasks 3, 6, 7, 8. ✓
- `normalize_text(raw: str) -> str` — defined Task 3, called in Task 7 and Task 8 with a `str` argument. ✓
- `extract_pdf_text(data: bytes) -> tuple[str, int]` — defined Task 6, unpacked as `raw_text, page_count` in Task 7. ✓
- `extract_docx_text(data: bytes) -> str` — defined Task 6, assigned to `raw_text` in Task 7. ✓
- `ResumeDoc` fields (`raw_text`, `source_format`, `filename`, `char_count`, `page_count`) — defined Task 4, constructed with exactly those keyword arguments in Task 7. ✓
- `JobDoc` fields (`raw_text`, `char_count`) — defined Task 4, constructed with exactly those keyword arguments in Task 8. ✓
- `__all__` in `ingestion/__init__.py` — set in Task 7, extended in Task 8; the final list matches the names tested in `test_ingest_resume.py` and `test_ingest_job.py`. ✓
- Fixture names (`resume_pdf_bytes`, `blank_pdf_bytes`, `resume_docx_bytes`, `resume_txt_bytes`) — defined Task 5, consumed by exactly those names in Tasks 6 and 7. ✓
