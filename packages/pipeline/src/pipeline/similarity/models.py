from __future__ import annotations

from pydantic import BaseModel, Field


class SkillMatch(BaseModel):
    """A single JD skill compared against the closest resume skill."""

    jd_skill: str = Field(description="The JD skill being matched.")
    resume_skill: str = Field(
        description="The closest resume skill (empty string when the resume has no skills).",
    )
    similarity: float = Field(ge=0.0, le=1.0, description="Cosine similarity in [0, 1].")
    matched: bool = Field(description="True if similarity >= matcher threshold.")


class SkillMatchReport(BaseModel):
    """Aggregate result of `SkillMatcher.match(resume, job)`."""

    required_matched: list[SkillMatch]
    required_missing: list[SkillMatch]
    nice_to_have_matched: list[SkillMatch]
    nice_to_have_missing: list[SkillMatch]
    match_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="matched-required / total-required. 1.0 when total-required == 0.",
    )
