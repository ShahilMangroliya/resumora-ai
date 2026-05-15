from typing import Literal

from pydantic import BaseModel, Field

Label = Literal["strong", "partial", "weak"]
Source = Literal["synthetic", "gold"]

LABEL_TO_SCORE: dict[Label, int] = {
    "strong": 85,
    "partial": 55,
    "weak": 20,
}


def score_for_label(label: Label) -> int:
    """Return the canonical bucket midpoint score for a label."""
    return LABEL_TO_SCORE[label]


class Pair(BaseModel):
    """One (resume, job description) training pair."""

    pair_id: str
    resume_text: str
    jd_text: str
    label: Label
    score: int = Field(ge=0, le=100)
    role: str
    seniority: str
    source: Source
    generator_model: str
    generated_at: str
    prompt_seed: int
