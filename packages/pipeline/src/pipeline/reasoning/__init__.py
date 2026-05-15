"""Phase 5 reasoning library: turn score + skill report + profiles into reasons and bullet rewrites."""

from pipeline.reasoning.errors import ReasoningError
from pipeline.reasoning.models import (
    REASON_CATEGORIES,
    BulletRewrite,
    Reason,
    ReasonCategory,
    ReasoningResult,
)
from pipeline.reasoning.reasoner import generate_reasoning

__all__ = [
    "REASON_CATEGORIES",
    "BulletRewrite",
    "Reason",
    "ReasonCategory",
    "ReasoningError",
    "ReasoningResult",
    "generate_reasoning",
]
