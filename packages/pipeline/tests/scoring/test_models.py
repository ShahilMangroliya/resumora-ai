import pytest
from pydantic import ValidationError

from pipeline.scoring.models import PREDICTED_LABELS, ScoreResult


def test_predicted_labels_constant():
    assert PREDICTED_LABELS == ("weak", "partial", "strong")


def test_score_result_valid():
    r = ScoreResult(
        score=72.3,
        confidence=0.81,
        class_probabilities={"weak": 0.05, "partial": 0.14, "strong": 0.81},
        predicted_label="strong",
    )
    assert r.score == 72.3
    assert r.predicted_label == "strong"


def test_score_result_score_below_20_rejected():
    with pytest.raises(ValidationError):
        ScoreResult(
            score=15.0,
            confidence=0.5,
            class_probabilities={"weak": 0.6, "partial": 0.3, "strong": 0.1},
            predicted_label="weak",
        )


def test_score_result_score_above_85_rejected():
    with pytest.raises(ValidationError):
        ScoreResult(
            score=90.0,
            confidence=0.5,
            class_probabilities={"weak": 0.1, "partial": 0.3, "strong": 0.6},
            predicted_label="strong",
        )


def test_score_result_invalid_label_rejected():
    with pytest.raises(ValidationError):
        ScoreResult(
            score=50.0,
            confidence=0.5,
            class_probabilities={"weak": 0.4, "partial": 0.4, "strong": 0.2},
            predicted_label="excellent",
        )


def test_score_result_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        ScoreResult(
            score=50.0,
            confidence=1.4,
            class_probabilities={"weak": 0.4, "partial": 0.4, "strong": 0.2},
            predicted_label="weak",
        )


def test_score_result_missing_class_key_rejected():
    with pytest.raises(ValidationError):
        ScoreResult(
            score=50.0,
            confidence=0.5,
            class_probabilities={"weak": 0.5, "partial": 0.5},
            predicted_label="weak",
        )
