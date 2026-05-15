from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

PredictedLabel = Literal["weak", "partial", "strong"]
PREDICTED_LABELS: tuple[PredictedLabel, ...] = ("weak", "partial", "strong")


class ScoreResult(BaseModel):
    """Output of `Scorer.score(resume_text, jd_text)`.

    `score` is bounded to [20.0, 85.0] because the underlying classifier emits
    a softmax-weighted average of bucket midpoints [20, 55, 85]. See the Phase 3
    supplement §1.1 for the rationale.
    """

    score: float = Field(
        ge=20.0,
        le=85.0,
        description="Expected-value fit score in [20.0, 85.0].",
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Max softmax probability.")
    class_probabilities: dict[str, float] = Field(
        description="Probabilities for {weak, partial, strong}; values in [0, 1] sum to ~1.",
    )
    predicted_label: PredictedLabel

    @field_validator("class_probabilities")
    @classmethod
    def _exact_three_keys(cls, value: dict[str, float]) -> dict[str, float]:
        expected = set(PREDICTED_LABELS)
        if set(value.keys()) != expected:
            raise ValueError(
                f"class_probabilities must have keys {sorted(expected)}; got {sorted(value.keys())}"
            )
        for k, v in value.items():
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"class_probabilities[{k!r}] = {v} not in [0, 1]")
        return value
