from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Protocol

from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download

from training.dataset.jsonl import read_pairs
from training.dataset.schema import Label, Pair

LABEL_TO_INT: dict[Label, int] = {"weak": 0, "partial": 1, "strong": 2}
INT_TO_LABEL: dict[int, Label] = {v: k for k, v in LABEL_TO_INT.items()}


class _Tokenizer(Protocol):
    """Anything that behaves like an HF tokenizer for sentence-pair input."""

    def __call__(self, text_a: str, text_b: str, truncation: bool, max_length: int) -> dict[str, Any]:
        ...


def load_pairs_local(path: Path) -> list[Pair]:
    """Read every pair from a local JSONL file (reuses Phase 2 IO)."""
    return read_pairs(path)


def load_pairs_from_hub(repo_id: str, filename: str, hf_token: str | None = None) -> list[Pair]:
    """Download a JSONL file from an HF *dataset* repo and parse it as pairs.

    `filename` is the path inside the repo, e.g. "synthetic/pairs.jsonl".
    """
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=hf_token,
    )
    return read_pairs(Path(local_path))


def assert_label_balance(pairs: list[Pair], max_deviation: float = 0.20) -> None:
    """Guard against silently-skewed training data.

    Per the supplement §6.3: every label must be within `max_deviation` of the
    perfect 1/3 share. A 60/20/20 dataset will train a degenerate classifier;
    the assertion forces it into the foreground before the run starts.
    """
    if not pairs:
        raise ValueError("cannot train on an empty pair set")
    counts = Counter(p.label for p in pairs)
    total = len(pairs)
    expected = 1 / 3
    for label, count in counts.items():
        share = count / total
        if abs(share - expected) > max_deviation:
            raise ValueError(
                f"label balance check failed: {label}={share:.0%} of {total} "
                f"(expected within {max_deviation:.0%} of {expected:.0%})"
            )


def build_dataset(
    pairs: list[Pair],
    *,
    tokenizer: _Tokenizer,
    max_length: int,
) -> Dataset:
    """Tokenize resume + JD as a sentence pair and attach the integer label."""
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        encoded = tokenizer(pair.resume_text, pair.jd_text, truncation=True, max_length=max_length)
        rows.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "label": LABEL_TO_INT[pair.label],
            }
        )
    return Dataset.from_list(rows)


def train_val_split(dataset: Dataset, *, val_fraction: float, seed: int) -> DatasetDict:
    """Stratified-ish random split. Returns DatasetDict with `train` and `validation`."""
    split = dataset.train_test_split(test_size=val_fraction, seed=seed)
    return DatasetDict({"train": split["train"], "validation": split["test"]})
