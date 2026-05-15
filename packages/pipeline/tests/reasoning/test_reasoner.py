from dataclasses import dataclass, field

import pytest

from pipeline.extraction.errors import ExtractionError
from pipeline.extraction.models import Experience, JobProfile, ResumeProfile
from pipeline.reasoning import (
    BulletRewrite,
    Reason,
    ReasoningError,
    ReasoningResult,
    generate_reasoning,
)
from pipeline.scoring.models import ScoreResult
from pipeline.similarity.models import SkillMatch, SkillMatchReport


@dataclass
class FakeClient:
    payloads: list[dict | Exception]
    seen_prompts: list[str] = field(default_factory=list)

    def generate_json(self, prompt: str) -> dict:
        self.seen_prompts.append(prompt)
        head = self.payloads.pop(0)
        if isinstance(head, Exception):
            raise head
        return head


def _score() -> ScoreResult:
    return ScoreResult(
        score=72.0,
        confidence=0.8,
        class_probabilities={"weak": 0.1, "partial": 0.1, "strong": 0.8},
        predicted_label="strong",
    )


def _report() -> SkillMatchReport:
    return SkillMatchReport(
        required_matched=[
            SkillMatch(jd_skill="python", resume_skill="Python", similarity=0.99, matched=True)
        ],
        required_missing=[],
        nice_to_have_matched=[],
        nice_to_have_missing=[],
        match_rate=1.0,
    )


def _resume_profile() -> ResumeProfile:
    return ResumeProfile(
        titles=["Backend Engineer"],
        skills=["python"],
        experiences=[Experience(title="Backend Engineer", years=3.0)],
        education=["BSc"],
        total_years_experience=3.0,
    )


def _job_profile() -> JobProfile:
    return JobProfile(
        title="Senior Backend Engineer",
        required_skills=["python"],
        nice_to_have_skills=[],
        seniority="senior",
        min_years_experience=5.0,
    )


def _valid_payload() -> dict:
    return {
        "reasons": [
            {"summary": "s1", "evidence": "e1", "category": "matched_skill"},
            {"summary": "s2", "evidence": "e2", "category": "missing_skill"},
            {"summary": "s3", "evidence": "e3", "category": "experience_gap"},
        ],
        "rewrites": [
            {"original": "o1", "rewritten": "r1", "rationale": "why1"},
            {"original": "o2", "rewritten": "r2", "rationale": "why2"},
            {"original": "", "rewritten": "r3", "rationale": "why3"},
        ],
    }


def test_generate_reasoning_returns_validated_result():
    client = FakeClient(payloads=[_valid_payload()])
    result = generate_reasoning(
        score_result=_score(),
        skill_report=_report(),
        resume_profile=_resume_profile(),
        job_profile=_job_profile(),
        resume_text="some resume text",
        client=client,
    )
    assert isinstance(result, ReasoningResult)
    assert len(result.reasons) == 3
    assert len(result.rewrites) == 3
    assert isinstance(result.reasons[0], Reason)
    assert isinstance(result.rewrites[2], BulletRewrite)
    assert result.reasons[1].category == "missing_skill"


def test_generate_reasoning_threads_resume_text_into_prompt():
    client = FakeClient(payloads=[_valid_payload()])
    generate_reasoning(
        score_result=_score(),
        skill_report=_report(),
        resume_profile=_resume_profile(),
        job_profile=_job_profile(),
        resume_text="UNIQUE-RESUME-MARKER-9182",
        client=client,
    )
    assert "UNIQUE-RESUME-MARKER-9182" in client.seen_prompts[0]


def test_generate_reasoning_raises_on_wrong_reason_count():
    bad = _valid_payload()
    bad["reasons"] = bad["reasons"][:2]
    client = FakeClient(payloads=[bad])
    with pytest.raises(ReasoningError, match="invalid"):
        generate_reasoning(
            score_result=_score(),
            skill_report=_report(),
            resume_profile=_resume_profile(),
            job_profile=_job_profile(),
            resume_text="x",
            client=client,
        )


def test_generate_reasoning_raises_on_wrong_rewrite_count():
    bad = _valid_payload()
    bad["rewrites"] = bad["rewrites"] + [{"original": "", "rewritten": "x", "rationale": "y"}]
    client = FakeClient(payloads=[bad])
    with pytest.raises(ReasoningError, match="invalid"):
        generate_reasoning(
            score_result=_score(),
            skill_report=_report(),
            resume_profile=_resume_profile(),
            job_profile=_job_profile(),
            resume_text="x",
            client=client,
        )


def test_generate_reasoning_raises_on_bad_category():
    bad = _valid_payload()
    bad["reasons"][0]["category"] = "excellent"
    client = FakeClient(payloads=[bad])
    with pytest.raises(ReasoningError, match="invalid"):
        generate_reasoning(
            score_result=_score(),
            skill_report=_report(),
            resume_profile=_resume_profile(),
            job_profile=_job_profile(),
            resume_text="x",
            client=client,
        )


def test_generate_reasoning_wraps_extraction_error_as_reasoning_error():
    client = FakeClient(payloads=[ExtractionError("Ollama unreachable: boom")])
    with pytest.raises(ReasoningError, match="Ollama"):
        generate_reasoning(
            score_result=_score(),
            skill_report=_report(),
            resume_profile=_resume_profile(),
            job_profile=_job_profile(),
            resume_text="x",
            client=client,
        )
