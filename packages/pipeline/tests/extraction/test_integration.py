import httpx
import pytest

from pipeline.extraction import (
    OllamaClient,
    extract_job_profile,
    extract_resume_profile,
)
from pipeline.ingestion.models import JobDoc, ResumeDoc


def _ollama_up(base_url: str = "http://localhost:11434") -> bool:
    try:
        return httpx.get(f"{base_url}/api/tags", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _ollama_up(), reason="Ollama server is not reachable"),
]


_RESUME = """Jane Doe
Senior Backend Engineer

Experience:
- Acme Corp — Senior Backend Engineer (2021–2025): Python, FastAPI, Postgres, AWS.
- Beta Inc — Backend Engineer (2018–2021): Django, Redis.

Education: BSc Computer Science, University X (2018).
"""

_JOB = """We are hiring a Senior Backend Engineer.

Requirements:
- 5+ years of Python experience
- FastAPI or Django
- Postgres / Redis
- Nice to have: AWS, Docker
"""


def test_resume_extraction_returns_skills_and_years():
    client = OllamaClient()
    doc = ResumeDoc(
        raw_text=_RESUME,
        source_format="txt",
        filename="r.txt",
        char_count=len(_RESUME),
    )
    profile = extract_resume_profile(doc, client=client)
    skills = {s.lower() for s in profile.skills}
    assert "python" in skills
    assert profile.total_years_experience >= 3


def test_job_extraction_returns_required_skills_and_seniority():
    client = OllamaClient()
    doc = JobDoc(raw_text=_JOB, char_count=len(_JOB))
    profile = extract_job_profile(doc, client=client)
    required = {s.lower() for s in profile.required_skills}
    assert "python" in required
    assert profile.seniority in {"senior", "staff", "principal"}
