"""Phase 4 similarity library: skill-level matching via sentence-transformer embeddings."""

from pipeline.similarity.matcher import SkillMatcher
from pipeline.similarity.models import SkillMatch, SkillMatchReport

__all__ = [
    "SkillMatch",
    "SkillMatchReport",
    "SkillMatcher",
]
