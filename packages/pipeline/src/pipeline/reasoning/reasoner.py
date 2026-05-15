from __future__ import annotations

from typing import Protocol

from pydantic import ValidationError

from pipeline.extraction.client import OllamaClient
from pipeline.extraction.errors import ExtractionError
from pipeline.extraction.models import JobProfile, ResumeProfile
from pipeline.reasoning.errors import ReasoningError
from pipeline.reasoning.models import ReasoningResult
from pipeline.reasoning.prompts import build_reasoning_prompt
from pipeline.scoring.models import ScoreResult
from pipeline.similarity.models import SkillMatchReport


class _Client(Protocol):
    def generate_json(self, prompt: str) -> dict: ...


def _default_client() -> _Client:
    return OllamaClient()


def generate_reasoning(
    *,
    score_result: ScoreResult,
    skill_report: SkillMatchReport,
    resume_profile: ResumeProfile,
    job_profile: JobProfile,
    resume_text: str,
    client: _Client | None = None,
) -> ReasoningResult:
    """Generate 3 reasons + 3 bullet rewrites for a scored (resume, JD) pair.

    Uses a local Ollama LLM via the injected `client`. The default client is
    `pipeline.extraction.client.OllamaClient` — deliberate cross-module reuse
    (see Phase 5 supplement §2.5). Pass a custom client to swap models.

    Raises `ReasoningError` on transport failure, unparseable JSON, or any
    payload that does not satisfy the `ReasoningResult` shape. The Phase 6
    API catches `ReasoningError` and degrades to a score-only response.
    """
    client = client or _default_client()
    prompt = build_reasoning_prompt(
        score_result=score_result,
        skill_report=skill_report,
        resume_profile=resume_profile,
        job_profile=job_profile,
        resume_text=resume_text,
    )
    try:
        payload = client.generate_json(prompt)
    except ExtractionError as exc:
        raise ReasoningError(str(exc)) from exc

    try:
        return ReasoningResult.model_validate(payload)
    except ValidationError as exc:
        raise ReasoningError(f"Ollama returned invalid ReasoningResult: {exc}") from exc
