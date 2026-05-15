import pytest
from pydantic import ValidationError

from pipeline.reasoning.models import (
    REASON_CATEGORIES,
    BulletRewrite,
    Reason,
    ReasoningResult,
)


def test_reason_categories_constant():
    assert REASON_CATEGORIES == (
        "matched_skill",
        "missing_skill",
        "experience_match",
        "experience_gap",
        "other",
    )


def test_reason_valid():
    r = Reason(
        summary="Strong Python skills match required stack",
        evidence="Resume lists 5 years of Python; JD requires senior Python.",
        category="matched_skill",
    )
    assert r.category == "matched_skill"


def test_reason_invalid_category_rejected():
    with pytest.raises(ValidationError):
        Reason(summary="x", evidence="y", category="excellent")


def test_bullet_rewrite_valid():
    b = BulletRewrite(
        original="- Built things with Python",
        rewritten="- Built and shipped 3 production services in Python (FastAPI, ~10k req/s).",
        rationale="Quantifies impact and names the framework the JD requires.",
    )
    assert b.rewritten.startswith("-")


def test_bullet_rewrite_empty_original_allowed_for_synthesis():
    b = BulletRewrite(original="", rewritten="- New synthesized bullet.", rationale="why")
    assert b.original == ""


def _three_reasons() -> list[Reason]:
    return [
        Reason(summary=f"Reason {i}", evidence=f"Evidence {i}", category="matched_skill")
        for i in range(3)
    ]


def _three_rewrites() -> list[BulletRewrite]:
    return [
        BulletRewrite(original=f"orig {i}", rewritten=f"rewritten {i}", rationale=f"r {i}")
        for i in range(3)
    ]


def test_reasoning_result_valid_with_strict_three():
    r = ReasoningResult(reasons=_three_reasons(), rewrites=_three_rewrites())
    assert len(r.reasons) == 3
    assert len(r.rewrites) == 3


def test_reasoning_result_rejects_fewer_than_three_reasons():
    with pytest.raises(ValidationError):
        ReasoningResult(reasons=_three_reasons()[:2], rewrites=_three_rewrites())


def test_reasoning_result_rejects_more_than_three_reasons():
    with pytest.raises(ValidationError):
        ReasoningResult(reasons=_three_reasons() + [_three_reasons()[0]], rewrites=_three_rewrites())


def test_reasoning_result_rejects_fewer_than_three_rewrites():
    with pytest.raises(ValidationError):
        ReasoningResult(reasons=_three_reasons(), rewrites=_three_rewrites()[:2])


def test_reasoning_result_rejects_more_than_three_rewrites():
    with pytest.raises(ValidationError):
        ReasoningResult(
            reasons=_three_reasons(), rewrites=_three_rewrites() + [_three_rewrites()[0]]
        )
