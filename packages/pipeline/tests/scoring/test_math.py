import numpy as np

from pipeline.scoring._math import (
    BUCKET_SCORES,
    INT_TO_LABEL,
    confidence_from_logits,
    score_from_logits,
    softmax,
)


def test_bucket_scores_match_phase3_supplement():
    # Pinned to the Phase 3 supplement §1: the actual range is [20, 85].
    assert BUCKET_SCORES == [20.0, 55.0, 85.0]


def test_int_to_label_mapping_is_canonical():
    assert INT_TO_LABEL == {0: "weak", 1: "partial", 2: "strong"}


def test_softmax_rows_sum_to_one():
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    probs = softmax(logits)
    np.testing.assert_allclose(probs.sum(axis=-1), [1.0, 1.0], atol=1e-7)


def test_score_from_logits_is_bounded_to_20_85():
    logits = np.array([
        [10.0, 0.0, 0.0],   # weak
        [0.0, 0.0, 10.0],   # strong
        [0.0, 10.0, 0.0],   # partial
    ])
    scores = score_from_logits(logits)
    assert scores[0] < 25
    assert scores[1] > 80
    assert 50 < scores[2] < 60
    for s in scores:
        assert 20.0 <= s <= 85.0


def test_score_from_logits_uniform_is_average_bucket():
    logits = np.zeros((1, 3))
    scores = score_from_logits(logits)
    assert abs(scores[0] - (20 + 55 + 85) / 3) < 1e-9


def test_confidence_from_logits_is_max_prob():
    logits = np.array([[10.0, 0.0, 0.0]])
    conf = confidence_from_logits(logits)
    assert conf[0] > 0.99


def test_score_from_logits_accepts_lists():
    scores = score_from_logits([[0.0, 0.0, 0.0]])
    assert abs(scores[0] - (20 + 55 + 85) / 3) < 1e-9
