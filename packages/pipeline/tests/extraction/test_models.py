import pytest
from pydantic import ValidationError

from pipeline.extraction.models import Experience, JobProfile, ResumeProfile


def test_experience_is_validated():
    exp = Experience(title="Senior Backend Engineer", years=4.5)
    assert exp.title == "Senior Backend Engineer"
    assert exp.years == 4.5


def test_experience_rejects_negative_years():
    with pytest.raises(ValidationError):
        Experience(title="dev", years=-1)


def test_resume_profile_minimum_fields():
    profile = ResumeProfile(
        titles=["Backend Engineer"],
        skills=["python", "fastapi"],
        experiences=[Experience(title="Backend Engineer", years=3.0)],
        education=["BSc Computer Science"],
        total_years_experience=3.0,
    )
    assert profile.skills == ["python", "fastapi"]
    assert profile.total_years_experience == 3.0


def test_job_profile_minimum_fields():
    profile = JobProfile(
        title="Backend Engineer",
        required_skills=["python"],
        nice_to_have_skills=["docker"],
        seniority="mid",
        min_years_experience=2.0,
    )
    assert profile.title == "Backend Engineer"
    assert profile.seniority == "mid"


def test_seniority_rejects_unknown_value():
    with pytest.raises(ValidationError):
        JobProfile(
            title="t",
            required_skills=[],
            nice_to_have_skills=[],
            seniority="emperor",
            min_years_experience=0,
        )
