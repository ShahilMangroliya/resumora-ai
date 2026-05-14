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
