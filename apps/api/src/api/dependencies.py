from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Protocol

from fastapi import Depends
from pipeline.extraction import (
    JobProfile,
    OllamaClient,
    ResumeProfile,
    extract_job_profile,
    extract_resume_profile,
)
from pipeline.ingestion.models import JobDoc, ResumeDoc
from pipeline.reasoning import ReasoningResult, generate_reasoning
from pipeline.scoring import Scorer
from pipeline.scoring.models import ScoreResult
from pipeline.similarity import SkillMatcher
from pipeline.similarity.models import SkillMatchReport

from api.config import load_settings


class _ClientProto(Protocol):
    def generate_json(self, prompt: str) -> dict: ...


ExtractResumeFn = Callable[..., ResumeProfile]
ExtractJobFn = Callable[..., JobProfile]
ReasoningFn = Callable[..., ReasoningResult]


@dataclass
class Pipeline:
    """Bundle of every dependency the orchestrator needs.

    `extract_resume_fn`, `extract_job_fn`, `reasoning_fn` default to the real
    pipeline functions; tests build a `Pipeline` with fakes and override
    `get_pipeline` to return it.
    """

    scorer: Scorer
    matcher: SkillMatcher
    ollama_client: _ClientProto
    extract_resume_fn: ExtractResumeFn = extract_resume_profile
    extract_job_fn: ExtractJobFn = extract_job_profile
    reasoning_fn: ReasoningFn = generate_reasoning


@lru_cache(maxsize=1)
def _scorer_singleton() -> Scorer:
    settings = load_settings()
    return Scorer.from_pretrained(settings.scorer_repo, device=settings.scorer_device)


@lru_cache(maxsize=1)
def _matcher_singleton() -> SkillMatcher:
    settings = load_settings()
    return SkillMatcher.from_pretrained(device=settings.matcher_device)


@lru_cache(maxsize=1)
def _ollama_singleton() -> OllamaClient:
    settings = load_settings()
    return OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout=settings.ollama_timeout,
    )


def reset_singletons() -> None:
    """Clear all cached singletons. Tests call this between env-var tweaks."""
    _scorer_singleton.cache_clear()
    _matcher_singleton.cache_clear()
    _ollama_singleton.cache_clear()


def get_scorer() -> Scorer:
    return _scorer_singleton()


def get_matcher() -> SkillMatcher:
    return _matcher_singleton()


def get_ollama_client() -> _ClientProto:
    return _ollama_singleton()


def get_pipeline(
    scorer: Scorer = Depends(get_scorer),
    matcher: SkillMatcher = Depends(get_matcher),
    ollama_client: _ClientProto = Depends(get_ollama_client),
) -> Pipeline:
    return Pipeline(scorer=scorer, matcher=matcher, ollama_client=ollama_client)


__all__ = [
    "Pipeline",
    "get_matcher",
    "get_ollama_client",
    "get_pipeline",
    "get_scorer",
    "reset_singletons",
    # re-exported for orchestrator type hints
    "JobDoc",
    "JobProfile",
    "ReasoningResult",
    "ResumeDoc",
    "ResumeProfile",
    "ScoreResult",
    "SkillMatchReport",
]
