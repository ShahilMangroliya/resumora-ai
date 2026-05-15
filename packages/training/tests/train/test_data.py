from pathlib import Path

import pytest

from training.dataset.jsonl import write_pairs
from training.dataset.schema import Pair
from training.train import data as data_mod


def _make_pair(pair_id: str, label: str, score: int) -> Pair:
    return Pair(
        pair_id=pair_id,
        resume_text="alice has 5 years of python experience",
        jd_text="we need a senior python engineer",
        label=label,
        score=score,
        role="backend_dev",
        seniority="senior",
        source="synthetic",
        generator_model="llama3.2:3b",
        generated_at="2026-05-15T00:00:00Z",
        prompt_seed=0,
    )


def test_label_to_int_round_trip():
    assert data_mod.LABEL_TO_INT["weak"] == 0
    assert data_mod.LABEL_TO_INT["partial"] == 1
    assert data_mod.LABEL_TO_INT["strong"] == 2
    assert data_mod.INT_TO_LABEL[0] == "weak"
    assert data_mod.INT_TO_LABEL[1] == "partial"
    assert data_mod.INT_TO_LABEL[2] == "strong"


def test_load_pairs_local_reads_jsonl(tmp_path: Path):
    path = tmp_path / "pairs.jsonl"
    write_pairs(path, [_make_pair("p1", "weak", 20), _make_pair("p2", "strong", 85)])
    pairs = data_mod.load_pairs_local(path)
    assert [p.pair_id for p in pairs] == ["p1", "p2"]


def test_assert_label_balance_passes_when_balanced():
    pairs = [_make_pair(f"p{i}", "weak", 20) for i in range(3)] + \
            [_make_pair(f"q{i}", "partial", 55) for i in range(3)] + \
            [_make_pair(f"r{i}", "strong", 85) for i in range(3)]
    data_mod.assert_label_balance(pairs)


def test_assert_label_balance_raises_when_skewed():
    pairs = [_make_pair(f"p{i}", "weak", 20) for i in range(10)] + \
            [_make_pair("q", "partial", 55), _make_pair("r", "strong", 85)]
    with pytest.raises(ValueError, match="balance"):
        data_mod.assert_label_balance(pairs)


def test_assert_label_balance_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        data_mod.assert_label_balance([])


def test_build_dataset_tokenizes_and_carries_label():
    pairs = [_make_pair("p1", "weak", 20), _make_pair("p2", "strong", 85)]
    tokenizer = _FakeTokenizer()
    ds = data_mod.build_dataset(pairs, tokenizer=tokenizer, max_length=16)
    assert set(ds.column_names) == {"input_ids", "attention_mask", "label"}
    assert ds[0]["label"] == data_mod.LABEL_TO_INT["weak"]
    assert ds[1]["label"] == data_mod.LABEL_TO_INT["strong"]
    assert len(ds[0]["input_ids"]) <= 16


def test_train_val_split_is_deterministic_for_same_seed():
    pairs = [_make_pair(f"p{i}", "weak", 20) for i in range(20)]
    tokenizer = _FakeTokenizer()
    ds = data_mod.build_dataset(pairs, tokenizer=tokenizer, max_length=16)
    split_a = data_mod.train_val_split(ds, val_fraction=0.2, seed=42)
    split_b = data_mod.train_val_split(ds, val_fraction=0.2, seed=42)
    assert split_a["train"]["label"] == split_b["train"]["label"]
    assert split_a["validation"]["label"] == split_b["validation"]["label"]


class _FakeTokenizer:
    """Minimal stand-in for an HF tokenizer.

    `build_dataset` only needs `tokenizer(text_a, text_b, truncation=..., max_length=...)`
    to return a dict with `input_ids` and `attention_mask` -- exactly what a real
    tokenizer does.
    """

    def __call__(self, text_a, text_b, truncation, max_length):
        ids = [1, 2, 3, 4][:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}
