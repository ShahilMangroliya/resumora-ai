"""Phase 4 scoring library: load the fine-tuned DistilBERT+LoRA model and score (resume, JD) pairs."""

from pipeline.scoring.models import PREDICTED_LABELS, PredictedLabel, ScoreResult
from pipeline.scoring.scorer import Scorer

__all__ = [
    "PREDICTED_LABELS",
    "PredictedLabel",
    "ScoreResult",
    "Scorer",
]
