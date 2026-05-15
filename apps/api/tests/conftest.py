from __future__ import annotations

import types
from dataclasses import dataclass

import pytest
from api.dependencies import reset_singletons
from pipeline.extraction.errors import ExtractionError
from pipeline.extraction.models import Experience, JobProfile, ResumeProfile
from pipeline.reasoning.errors import ReasoningError
from pipeline.reasoning.models import BulletRewrite, Reason, ReasoningResult
from pipeline.scoring.models import ScoreResult
from pipeline.similarity.models import SkillMatch, SkillMatchReport


@pytest.fixture(autouse=True)
def _clear_singletons_between_tests():
    """Reset lru_cache singletons so direct getter calls cannot leak state across tests."""
    reset_singletons()
    yield
    reset_singletons()

_RESUME_TEXT = (
    "Jane Doe\n"
    "Senior Python Engineer\n"
    "Skills: Python, FastAPI, Docker\n"
    "- Built a FastAPI service handling 5k req/s\n"
    "- Containerized 6 services with Docker\n"
)

_JD_TEXT = (
    "Senior Backend Engineer\n"
    "We need someone with Python, FastAPI, and Kubernetes experience.\n"
    "Nice to have: Docker, AWS.\n"
)


@pytest.fixture
def resume_txt_bytes() -> bytes:
    return _RESUME_TEXT.encode("utf-8")


@pytest.fixture
def jd_text() -> str:
    return _JD_TEXT


@pytest.fixture
def sample_score_result() -> ScoreResult:
    return ScoreResult(
        score=72.5,
        confidence=0.84,
        class_probabilities={"weak": 0.05, "partial": 0.11, "strong": 0.84},
        predicted_label="strong",
    )


@pytest.fixture
def sample_resume_profile() -> ResumeProfile:
    return ResumeProfile(
        titles=["Senior Python Engineer"],
        skills=["python", "fastapi", "docker"],
        experiences=[Experience(title="Senior Python Engineer", years=5.0)],
        education=["BSc Computer Science"],
        total_years_experience=5.0,
    )


@pytest.fixture
def sample_job_profile() -> JobProfile:
    return JobProfile(
        title="Senior Backend Engineer",
        required_skills=["python", "fastapi", "kubernetes"],
        nice_to_have_skills=["docker", "aws"],
        seniority="senior",
        min_years_experience=5.0,
    )


@pytest.fixture
def sample_skill_report() -> SkillMatchReport:
    return SkillMatchReport(
        required_matched=[
            SkillMatch(jd_skill="python", resume_skill="python", similarity=0.99, matched=True),
            SkillMatch(jd_skill="fastapi", resume_skill="fastapi", similarity=0.99, matched=True),
        ],
        required_missing=[
            SkillMatch(jd_skill="kubernetes", resume_skill="docker", similarity=0.4, matched=False),
        ],
        nice_to_have_matched=[
            SkillMatch(jd_skill="docker", resume_skill="docker", similarity=0.99, matched=True),
        ],
        nice_to_have_missing=[
            SkillMatch(jd_skill="aws", resume_skill="docker", similarity=0.3, matched=False),
        ],
        match_rate=2 / 3,
    )


@pytest.fixture
def sample_reasoning_result() -> ReasoningResult:
    return ReasoningResult(
        reasons=[
            Reason(summary="Strong Python match", evidence="5y Python", category="matched_skill"),
            Reason(summary="Missing Kubernetes", evidence="JD requires k8s", category="missing_skill"),
            Reason(summary="Seniority fit", evidence="5y matches min", category="experience_match"),
        ],
        rewrites=[
            BulletRewrite(original="- Built FastAPI service", rewritten="- Built and operated a FastAPI service handling 5k req/s in production", rationale="adds metric"),
            BulletRewrite(original="- Containerized 6 services with Docker", rewritten="- Containerized 6 microservices with Docker and deployed via Kubernetes", rationale="ties to JD"),
            BulletRewrite(original="", rewritten="- Owned on-call rotation across 3 backend services", rationale="signals seniority"),
        ],
    )


@dataclass
class FakeScorer:
    next_result: ScoreResult

    def score(self, resume_text: str, jd_text: str) -> ScoreResult:
        return self.next_result


@dataclass
class FakeMatcher:
    next_report: SkillMatchReport

    def match(self, resume_profile, job_profile) -> SkillMatchReport:
        return self.next_report


@dataclass
class FakeOllama:
    """Stand-in for OllamaClient; never called when extract/reason are stubbed."""

    def generate_json(self, prompt: str) -> dict:
        raise AssertionError("FakeOllama.generate_json should not be invoked in unit tests")


def _ok_extract_resume(profile: ResumeProfile):
    def _fn(doc, **_kw) -> ResumeProfile:
        return profile

    return _fn


def _ok_extract_job(profile: JobProfile):
    def _fn(doc, **_kw) -> JobProfile:
        return profile

    return _fn


def _ok_reasoning(result: ReasoningResult):
    def _fn(**_kw) -> ReasoningResult:
        return result

    return _fn


def _failing_extract_resume(exc: ExtractionError):
    def _fn(doc, **_kw) -> ResumeProfile:
        raise exc

    return _fn


def _failing_reasoning(exc: ReasoningError):
    def _fn(**_kw) -> ReasoningResult:
        raise exc

    return _fn


@pytest.fixture
def fakes_module():
    """Expose the helpers as attributes for ergonomic access in tests."""
    m = types.SimpleNamespace(
        FakeScorer=FakeScorer,
        FakeMatcher=FakeMatcher,
        FakeOllama=FakeOllama,
        ok_extract_resume=_ok_extract_resume,
        ok_extract_job=_ok_extract_job,
        ok_reasoning=_ok_reasoning,
        failing_extract_resume=_failing_extract_resume,
        failing_reasoning=_failing_reasoning,
    )
    return m
