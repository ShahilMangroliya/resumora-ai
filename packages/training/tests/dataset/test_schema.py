import pytest
from pydantic import ValidationError

from training.dataset.schema import LABEL_TO_SCORE, Pair, score_for_label


def test_score_for_label_returns_bucket_midpoint():
    assert score_for_label("strong") == 85
    assert score_for_label("partial") == 55
    assert score_for_label("weak") == 20


def test_label_to_score_is_complete():
    assert set(LABEL_TO_SCORE.keys()) == {"strong", "partial", "weak"}


def test_pair_round_trips_through_json():
    pair = Pair(
        pair_id="pair-0001",
        resume_text="r",
        jd_text="j",
        label="strong",
        score=85,
        role="backend-dev",
        seniority="mid",
        source="synthetic",
        generator_model="llama3.2:3b",
        generated_at="2026-05-15T10:00:00Z",
        prompt_seed=42,
    )
    raw = pair.model_dump_json()
    revived = Pair.model_validate_json(raw)
    assert revived == pair


def test_pair_rejects_unknown_label():
    with pytest.raises(ValidationError):
        Pair(
            pair_id="x",
            resume_text="r",
            jd_text="j",
            label="amazing",
            score=85,
            role="backend-dev",
            seniority="mid",
            source="synthetic",
            generator_model="llama3.2:3b",
            generated_at="2026-05-15T10:00:00Z",
            prompt_seed=0,
        )


def test_pair_rejects_score_outside_0_100():
    with pytest.raises(ValidationError):
        Pair(
            pair_id="x",
            resume_text="r",
            jd_text="j",
            label="strong",
            score=120,
            role="r",
            seniority="mid",
            source="synthetic",
            generator_model="m",
            generated_at="t",
            prompt_seed=0,
        )
