from __future__ import annotations

import numpy as np

# Pinned to the Phase 3 supplement §1. Re-declared (not imported from training)
# so pipeline stays training-independent at runtime; tests pin the values.
BUCKET_SCORES: list[float] = [20.0, 55.0, 85.0]
INT_TO_LABEL: dict[int, str] = {0: "weak", 1: "partial", 2: "strong"}

_SCORE_VECTOR = np.array(BUCKET_SCORES)


def softmax(logits: np.ndarray | list) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64)
    shifted = arr - arr.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def score_from_logits(logits: np.ndarray | list) -> np.ndarray:
    """Expected-value score: softmax(logits) · [20, 55, 85].

    Output is bounded to [20.0, 85.0] — the deliberate Phase 3 range
    (see Phase 3 supplement §1.1). The product-surface "0–100" is honored
    by disclosure, not by stretching the range.
    """
    probs = softmax(logits)
    return probs @ _SCORE_VECTOR


def confidence_from_logits(logits: np.ndarray | list) -> np.ndarray:
    """Max softmax probability — a simple per-prediction confidence in [0, 1]."""
    probs = softmax(logits)
    return probs.max(axis=-1)
