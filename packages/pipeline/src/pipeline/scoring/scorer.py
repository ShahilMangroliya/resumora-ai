from __future__ import annotations

from typing import Self

import numpy as np
import torch

from pipeline.scoring._math import (
    INT_TO_LABEL,
    confidence_from_logits,
    score_from_logits,
    softmax,
)
from pipeline.scoring.loader import load_scorer_artifacts
from pipeline.scoring.models import ScoreResult


class Scorer:
    """Eagerly-loaded inference wrapper for the Phase 3 DistilBERT+LoRA model.

    Score range is bounded to [20.0, 85.0] — see ScoreResult.score and the
    Phase 3 supplement §1.1.
    """

    def __init__(self, *, model, tokenizer, device: str, max_length: int = 512) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._max_length = max_length

    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_path: str,
        *,
        base_model: str = "distilbert-base-uncased",
        device: str = "cpu",
        max_length: int = 512,
    ) -> Self:
        """Load model + tokenizer eagerly and return a ready-to-use Scorer."""
        model, tokenizer = load_scorer_artifacts(
            repo_id_or_path=repo_id_or_path,
            base_model=base_model,
            device=device,
        )
        return cls(model=model, tokenizer=tokenizer, device=device, max_length=max_length)

    def score(self, resume_text: str, jd_text: str) -> ScoreResult:
        """Classify a (resume, JD) pair and return score + probabilities."""
        enc = self._tokenizer(
            resume_text,
            jd_text,
            truncation=True,
            max_length=self._max_length,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            out = self._model(**enc)
        logits_np = out.logits.detach().cpu().numpy()  # shape (1, 3)

        probs = softmax(logits_np)[0]
        score = float(score_from_logits(logits_np)[0])
        confidence = float(confidence_from_logits(logits_np)[0])
        pred_int = int(np.argmax(probs))

        return ScoreResult(
            score=score,
            confidence=confidence,
            class_probabilities={INT_TO_LABEL[i]: float(probs[i]) for i in range(3)},
            predicted_label=INT_TO_LABEL[pred_int],  # type: ignore[arg-type]
        )
