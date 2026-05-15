from __future__ import annotations

from pydantic import BaseModel, Field

from pipeline.reasoning.models import ReasoningResult
from pipeline.scoring.models import ScoreResult
from pipeline.similarity.models import SkillMatchReport


class AnalyzeResponse(BaseModel):
    """JSON body of a successful POST /analyze.

    `skill_report` and `reasoning` are nullable: when extraction fails (Ollama
    unreachable), the API still returns the score plus a human-readable
    `warnings` entry explaining which downstream stages were skipped.
    """

    score: ScoreResult
    skill_report: SkillMatchReport | None = Field(
        default=None,
        description="Skill-match report; null when extraction failed.",
    )
    reasoning: ReasoningResult | None = Field(
        default=None,
        description="Reasons + bullet rewrites; null when reasoning or upstream failed.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Human-readable notes about partial-result degradation.",
    )
