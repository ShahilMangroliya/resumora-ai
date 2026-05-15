from dataclasses import dataclass

import pytest

from pipeline.extraction import (
    ExtractionError,
    JobProfile,
    ResumeProfile,
    extract_job_profile,
    extract_resume_profile,
)
from pipeline.ingestion.models import JobDoc, ResumeDoc


@dataclass
class FakeClient:
    """Stands in for OllamaClient — returns canned JSON for each call."""

    payloads: list[dict]
    seen_prompts: list[str] | None = None

    def __post_init__(self):
        if self.seen_prompts is None:
            self.seen_prompts = []

    def generate_json(self, prompt: str) -> dict:
        self.seen_prompts.append(prompt)
        return self.payloads.pop(0)


def _resume_doc(text: str = "Jane Doe\nBackend Engineer") -> ResumeDoc:
    return ResumeDoc(
        raw_text=text,
        source_format="txt",
        filename="r.txt",
        char_count=len(text),
    )


def _job_doc(text: str = "Hiring a backend engineer") -> JobDoc:
    return JobDoc(raw_text=text, char_count=len(text))


def test_extract_resume_profile_returns_validated_model():
    client = FakeClient(
        payloads=[
            {
                "titles": ["Backend Engineer"],
                "skills": ["python"],
                "experiences": [{"title": "Backend Engineer", "years": 3}],
                "education": ["BSc"],
                "total_years_experience": 3,
            }
        ]
    )
    profile = extract_resume_profile(_resume_doc(), client=client)
    assert isinstance(profile, ResumeProfile)
    assert profile.skills == ["python"]


def test_extract_resume_profile_passes_text_to_prompt():
    client = FakeClient(
        payloads=[
            {
                "titles": [],
                "skills": [],
                "experiences": [],
                "education": [],
                "total_years_experience": 0,
            }
        ]
    )
    extract_resume_profile(_resume_doc("UNIQUE-TOKEN-12345"), client=client)
    assert "UNIQUE-TOKEN-12345" in client.seen_prompts[0]


def test_extract_resume_profile_raises_on_invalid_payload():
    client = FakeClient(payloads=[{"skills": "not-a-list"}])
    with pytest.raises(ExtractionError, match="invalid"):
        extract_resume_profile(_resume_doc(), client=client)


def test_extract_job_profile_returns_validated_model():
    client = FakeClient(
        payloads=[
            {
                "title": "Backend Engineer",
                "required_skills": ["python"],
                "nice_to_have_skills": ["docker"],
                "seniority": "mid",
                "min_years_experience": 2,
            }
        ]
    )
    profile = extract_job_profile(_job_doc(), client=client)
    assert isinstance(profile, JobProfile)
    assert profile.seniority == "mid"


def test_extract_job_profile_raises_on_invalid_payload():
    client = FakeClient(
        payloads=[
            {
                "title": "x",
                "required_skills": [],
                "nice_to_have_skills": [],
                "seniority": "emperor",
                "min_years_experience": 0,
            }
        ]
    )
    with pytest.raises(ExtractionError, match="invalid"):
        extract_job_profile(_job_doc(), client=client)
