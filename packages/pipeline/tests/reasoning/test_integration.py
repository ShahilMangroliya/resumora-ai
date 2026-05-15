import pytest

from pipeline.extraction.models import Experience, JobProfile, ResumeProfile
from pipeline.reasoning import ReasoningResult, generate_reasoning
from pipeline.scoring.models import ScoreResult
from pipeline.similarity.models import SkillMatch, SkillMatchReport


@pytest.mark.integration
def test_generate_reasoning_against_live_ollama():
    """Calls a local Ollama. Skipped unless `pytest -m integration` is used."""
    score = ScoreResult(
        score=68.0,
        confidence=0.7,
        class_probabilities={"weak": 0.1, "partial": 0.2, "strong": 0.7},
        predicted_label="strong",
    )
    report = SkillMatchReport(
        required_matched=[
            SkillMatch(jd_skill="python", resume_skill="Python", similarity=0.98, matched=True),
        ],
        required_missing=[
            SkillMatch(
                jd_skill="kubernetes", resume_skill="docker", similarity=0.4, matched=False
            ),
        ],
        nice_to_have_matched=[],
        nice_to_have_missing=[],
        match_rate=0.5,
    )
    resume_profile = ResumeProfile(
        titles=["Backend Engineer"],
        skills=["python", "fastapi", "docker"],
        experiences=[Experience(title="Backend Engineer", years=3.0)],
        education=["BSc Computer Science"],
        total_years_experience=3.0,
    )
    job_profile = JobProfile(
        title="Senior Backend Engineer",
        required_skills=["python", "kubernetes"],
        nice_to_have_skills=["aws"],
        seniority="senior",
        min_years_experience=5.0,
    )
    resume_text = (
        "Backend Engineer with 3 years of Python and FastAPI experience.\n"
        "- Built and shipped a real-time analytics service in Python\n"
        "- Containerized 8 services with Docker and deployed to AWS\n"
        "- Wrote integration tests with pytest at 90% coverage\n"
    )
    result = generate_reasoning(
        score_result=score,
        skill_report=report,
        resume_profile=resume_profile,
        job_profile=job_profile,
        resume_text=resume_text,
    )
    assert isinstance(result, ReasoningResult)
    assert len(result.reasons) == 3
    assert len(result.rewrites) == 3
    for r in result.reasons:
        assert r.summary
        assert r.category in (
            "matched_skill",
            "missing_skill",
            "experience_match",
            "experience_gap",
            "other",
        )
    for b in result.rewrites:
        assert b.rewritten
