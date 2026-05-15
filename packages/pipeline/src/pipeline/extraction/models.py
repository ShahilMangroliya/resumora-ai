from typing import Literal

from pydantic import BaseModel, Field

Seniority = Literal["intern", "junior", "mid", "senior", "staff", "principal"]


class Experience(BaseModel):
    """A single role on a resume, reduced to title + total years."""

    title: str
    years: float = Field(ge=0)


class ResumeProfile(BaseModel):
    """Structured fields extracted from a resume."""

    titles: list[str]
    skills: list[str]
    experiences: list[Experience]
    education: list[str]
    total_years_experience: float = Field(ge=0)


class JobProfile(BaseModel):
    """Structured fields extracted from a job description."""

    title: str
    required_skills: list[str]
    nice_to_have_skills: list[str]
    seniority: Seniority
    min_years_experience: float = Field(ge=0)
