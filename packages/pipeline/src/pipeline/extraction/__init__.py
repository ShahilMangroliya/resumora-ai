from __future__ import annotations

from typing import Protocol

from pydantic import ValidationError

from pipeline.extraction.client import OllamaClient
from pipeline.extraction.errors import ExtractionError
from pipeline.extraction.models import Experience, JobProfile, ResumeProfile
from pipeline.extraction.prompts import build_job_prompt, build_resume_prompt
from pipeline.ingestion.models import JobDoc, ResumeDoc

__all__ = [
    "Experience",
    "ExtractionError",
    "JobProfile",
    "OllamaClient",
    "ResumeProfile",
    "extract_job_profile",
    "extract_resume_profile",
]


class _Client(Protocol):
    def generate_json(self, prompt: str) -> dict: ...


def _default_client() -> _Client:
    return OllamaClient()


def extract_resume_profile(
    doc: ResumeDoc,
    *,
    client: _Client | None = None,
) -> ResumeProfile:
    """Extract a ResumeProfile from a normalized ResumeDoc via Ollama."""
    client = client or _default_client()
    payload = client.generate_json(build_resume_prompt(doc.raw_text))
    try:
        return ResumeProfile.model_validate(payload)
    except ValidationError as exc:
        raise ExtractionError(f"Ollama returned invalid ResumeProfile: {exc}") from exc


def extract_job_profile(
    doc: JobDoc,
    *,
    client: _Client | None = None,
) -> JobProfile:
    """Extract a JobProfile from a normalized JobDoc via Ollama."""
    client = client or _default_client()
    payload = client.generate_json(build_job_prompt(doc.raw_text))
    try:
        return JobProfile.model_validate(payload)
    except ValidationError as exc:
        raise ExtractionError(f"Ollama returned invalid JobProfile: {exc}") from exc
