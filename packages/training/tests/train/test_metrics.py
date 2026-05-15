import numpy as np

from training.train.metrics import (
    INT_TO_SCORE,
    compute_metrics,
    confidence_from_logits,
    score_from_logits,
)


def test_int_to_score_matches_bucket_midpoints():
    assert INT_TO_SCORE == {0: 20, 1: 55, 2: 85}


def test_score_from_logits_is_in_bucket_range():
    # Very confident "strong" → score near 85; very confident "weak" → score near 20.
    logits = np.array([[10.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
    scores = score_from_logits(logits)
    assert scores[0] < 25
    assert scores[1] > 80
    # All scores must land inside [20, 85].
    for s in scores:
        assert 20 <= s <= 85


def test_score_from_logits_uniform_lands_at_average_bucket():
    # Equal logits → uniform probs → score = (20 + 55 + 85) / 3 = 53.33...
    logits = np.zeros((1, 3))
    scores = score_from_logits(logits)
    assert abs(scores[0] - (20 + 55 + 85) / 3) < 1e-6


def test_confidence_from_logits_returns_max_prob():
    logits = np.array([[10.0, 0.0, 0.0]])
    conf = confidence_from_logits(logits)
    assert conf[0] > 0.99


def test_compute_metrics_basic_shape():
    # Perfect predictions for a tiny eval set.
    logits = np.array([
        [10.0, 0.0, 0.0],    # predicts weak
        [0.0, 10.0, 0.0],    # predicts partial
        [0.0, 0.0, 10.0],    # predicts strong
    ])
    labels = np.array([0, 1, 2])
    metrics = compute_metrics((logits, labels))
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["f1_weak"] == 1.0
    assert metrics["f1_partial"] == 1.0
    assert metrics["f1_strong"] == 1.0
    # Predictions equal gold → MAE is small (just softmax slack).
    assert metrics["mae"] < 5.0


def test_compute_metrics_handles_imperfect_predictions():
    # Two correct, one off-by-one (predicted partial but label was weak).
    logits = np.array([
        [0.0, 10.0, 0.0],    # predicts partial, label weak
        [0.0, 10.0, 0.0],    # predicts partial, label partial
        [0.0, 0.0, 10.0],    # predicts strong, label strong
    ])
    labels = np.array([0, 1, 2])
    metrics = compute_metrics((logits, labels))
    assert metrics["accuracy"] == 2 / 3
    assert metrics["macro_f1"] < 1.0
    # First row: predicted score ≈ 55, true bucket score = 20 → ~35 error.
    assert metrics["mae"] > 10
