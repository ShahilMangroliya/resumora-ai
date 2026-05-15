from pathlib import Path

import torch
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)

from pipeline.scoring import Scorer, ScoreResult


def _build_deterministic_scorer(tmp_path: Path, bias_to_class: int) -> Scorer:
    """Save a tiny DistilBERT whose classifier is biased to a chosen class.

    Setting the classifier bias to a large value for one class guarantees that
    every input is predicted as that class — useful for asserting Scorer behavior
    without relying on the random init of a tiny model.
    """
    cfg = DistilBertConfig(
        vocab_size=200,
        max_position_embeddings=64,
        num_hidden_layers=1,
        n_layers=1,
        n_heads=2,
        hidden_size=32,
        dim=32,
        hidden_dim=64,
        num_labels=3,
    )
    model = DistilBertForSequenceClassification(cfg)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.zero_()
        model.classifier.bias[bias_to_class] = 50.0
    out = tmp_path / "tiny"
    model.save_pretrained(out)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"tok{i}" for i in range(195)]
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tok = DistilBertTokenizerFast(vocab_file=str(vocab_path))
    tok.save_pretrained(out)
    # max_length=64 to match the tiny model's max_position_embeddings.
    return Scorer.from_pretrained(
        repo_id_or_path=str(out), base_model=str(out), device="cpu", max_length=64
    )


def test_scorer_returns_score_result_with_strong_bias(tmp_path: Path):
    scorer = _build_deterministic_scorer(tmp_path, bias_to_class=2)
    result = scorer.score("alice has 5 years of python", "we need a senior python engineer")
    assert isinstance(result, ScoreResult)
    assert result.predicted_label == "strong"
    assert result.score > 80
    assert result.confidence > 0.99
    total = sum(result.class_probabilities.values())
    assert abs(total - 1.0) < 1e-5


def test_scorer_returns_score_result_with_weak_bias(tmp_path: Path):
    scorer = _build_deterministic_scorer(tmp_path, bias_to_class=0)
    result = scorer.score("alice studied english", "we need a senior python engineer")
    assert result.predicted_label == "weak"
    assert result.score < 25
    assert result.confidence > 0.99


def test_scorer_max_length_param_truncates(tmp_path: Path):
    scorer = _build_deterministic_scorer(tmp_path, bias_to_class=1)
    long_text = "tok1 " * 1000
    result = scorer.score(long_text, long_text)
    assert isinstance(result, ScoreResult)


def test_scorer_is_repeatable(tmp_path: Path):
    """No randomness — same input → same output (model is in eval mode)."""
    scorer = _build_deterministic_scorer(tmp_path, bias_to_class=2)
    r1 = scorer.score("a", "b")
    r2 = scorer.score("a", "b")
    assert r1.score == r2.score
    assert r1.class_probabilities == r2.class_probabilities


def test_scoring_public_surface():
    """Both names are re-exported from the package root."""
    from pipeline.scoring import Scorer as S, ScoreResult as R

    assert S is Scorer
    assert R is ScoreResult
