from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

INT_TO_SCORE: dict[int, int] = {0: 20, 1: 55, 2: 85}
_SCORE_VECTOR = np.array([20.0, 55.0, 85.0])


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def score_from_logits(logits: np.ndarray) -> np.ndarray:
    """Expected-value score: softmax(logits) dotted with [20, 55, 85].

    Output is bounded to [20, 85]. The 0–100 product surface is honored by
    disclosure (model card §7.1 of the design supplement), not by stretching
    the range.
    """
    probs = _softmax(np.asarray(logits, dtype=np.float64))
    return probs @ _SCORE_VECTOR


def confidence_from_logits(logits: np.ndarray) -> np.ndarray:
    """Max softmax probability — a simple per-prediction confidence."""
    probs = _softmax(np.asarray(logits, dtype=np.float64))
    return probs.max(axis=-1)


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """HF Trainer `compute_metrics` hook.

    Reports accuracy, macro-F1, per-class F1, and MAE of the expected-value
    score against the bucket midpoint of the true label.
    """
    logits, labels = eval_pred
    logits = np.asarray(logits)
    labels = np.asarray(labels)

    preds = logits.argmax(axis=-1)
    accuracy = float((preds == labels).mean())

    macro_f1 = float(f1_score(labels, preds, average="macro", labels=[0, 1, 2], zero_division=0))
    per_class = f1_score(labels, preds, average=None, labels=[0, 1, 2], zero_division=0)

    pred_scores = score_from_logits(logits)
    true_scores = np.array([INT_TO_SCORE[int(label)] for label in labels])
    mae = float(np.abs(pred_scores - true_scores).mean())

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "f1_weak": float(per_class[0]),
        "f1_partial": float(per_class[1]),
        "f1_strong": float(per_class[2]),
        "mae": mae,
    }
