from pathlib import Path

import pytest
import torch
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)

from pipeline.scoring.loader import load_scorer_artifacts


def _make_tiny_base(tmp_path: Path) -> Path:
    """Build a tiny DistilBERT-for-seq-classification and save it to disk.

    The tokenizer is hand-built from a 200-token vocab so no download is needed.
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
    out = tmp_path / "tiny-base"
    model.save_pretrained(out)

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"tok{i}" for i in range(195)]
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tok = DistilBertTokenizerFast(vocab_file=str(vocab_path))
    tok.save_pretrained(out)
    return out


def test_load_scorer_artifacts_local_no_adapter(tmp_path: Path):
    base_dir = _make_tiny_base(tmp_path)
    # When repo_id_or_path == base_model with no adapter_config.json, the loader
    # falls through to loading repo_id_or_path as a standalone model.
    model, tokenizer = load_scorer_artifacts(
        repo_id_or_path=str(base_dir),
        base_model=str(base_dir),
        device="cpu",
    )
    assert hasattr(model, "forward")
    assert not model.training
    enc = tokenizer("hello", "world", truncation=True, max_length=16, return_tensors="pt")
    assert "input_ids" in enc
    with torch.no_grad():
        out = model(**enc)
    assert out.logits.shape[-1] == 3


def test_load_scorer_artifacts_falls_back_to_base_tokenizer(tmp_path: Path):
    base_dir = _make_tiny_base(tmp_path)
    adapter_dir = tmp_path / "adapter-no-tokenizer"
    adapter_dir.mkdir()
    for name in ("config.json", "model.safetensors", "pytorch_model.bin"):
        src = base_dir / name
        if src.exists():
            (adapter_dir / name).write_bytes(src.read_bytes())

    model, tokenizer = load_scorer_artifacts(
        repo_id_or_path=str(adapter_dir),
        base_model=str(base_dir),
        device="cpu",
    )
    assert tokenizer is not None
    enc = tokenizer("hello", "world", truncation=True, max_length=16, return_tensors="pt")
    assert "input_ids" in enc


@pytest.mark.integration
def test_load_scorer_artifacts_from_hub():
    """Smoke-load the published Phase 3 model.

    Requires `RESUMORA_AI_SCORER_REPO` env var pointing at the published model.
    Skipped by default; run with `pytest -m integration`.
    """
    import os

    repo = os.environ.get("RESUMORA_AI_SCORER_REPO")
    if not repo:
        pytest.skip("RESUMORA_AI_SCORER_REPO not set")
    model, tokenizer = load_scorer_artifacts(
        repo_id_or_path=repo,
        base_model="distilbert-base-uncased",
        device="cpu",
    )
    assert model is not None
    assert tokenizer is not None
