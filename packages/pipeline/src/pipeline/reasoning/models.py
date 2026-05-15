from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ReasonCategory = Literal[
    "matched_skill",
    "missing_skill",
    "experience_match",
    "experience_gap",
    "other",
]

REASON_CATEGORIES: tuple[ReasonCategory, ...] = (
    "matched_skill",
    "missing_skill",
    "experience_match",
    "experience_gap",
    "other",
)


class Reason(BaseModel):
    """One of the top-3 reasons explaining the fit score."""

    summary: str = Field(description="Short, concrete one-line reason.")
    evidence: str = Field(description="Short citation from the resume or JD.")
    category: ReasonCategory


class BulletRewrite(BaseModel):
    """A suggested rewrite of a resume bullet for a specific JD."""

    original: str = Field(
        description="The original bullet from the resume. Empty string when synthesized.",
    )
    rewritten: str = Field(description="The suggested rewrite, tuned for the JD.")
    rationale: str = Field(description="Why this rewrite improves the fit.")


class ReasoningResult(BaseModel):
    """Output of `generate_reasoning(...)` — exactly 3 reasons and 3 rewrites."""

    reasons: list[Reason] = Field(
        min_length=3,
        max_length=3,
        description="Exactly three reasons.",
    )
    rewrites: list[BulletRewrite] = Field(
        min_length=3,
        max_length=3,
        description="Exactly three bullet rewrites.",
    )
