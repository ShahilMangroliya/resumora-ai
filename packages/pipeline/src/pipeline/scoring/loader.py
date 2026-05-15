from __future__ import annotations

from pathlib import Path

from peft import PeftConfig, PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _has_peft_adapter(repo_id_or_path: str) -> bool:
    """Detect a PEFT adapter at a local path. Hub lookups are best-effort."""
    p = Path(repo_id_or_path)
    if p.exists():
        return (p / "adapter_config.json").exists()
    try:
        PeftConfig.from_pretrained(repo_id_or_path)
        return True
    except Exception:  # noqa: BLE001
        return False


def _load_tokenizer(repo_id_or_path: str, fallback_base_model: str):
    """Try the adapter repo first; fall back to the base model's tokenizer."""
    try:
        return AutoTokenizer.from_pretrained(repo_id_or_path)
    except (OSError, ValueError):
        return AutoTokenizer.from_pretrained(fallback_base_model)


def load_scorer_artifacts(
    *,
    repo_id_or_path: str,
    base_model: str = "distilbert-base-uncased",
    device: str = "cpu",
):
    """Load the (model, tokenizer) pair for inference.

    `repo_id_or_path` may be a Hub repo (`USER/model-name`) or a local directory.
    If a PEFT adapter is detected, the base model is loaded first and the adapter
    is applied on top; otherwise `repo_id_or_path` is loaded as a standalone model
    (this path is mostly used in tests).
    """
    tokenizer = _load_tokenizer(repo_id_or_path, base_model)

    if _has_peft_adapter(repo_id_or_path):
        base = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=3)
        model = PeftModel.from_pretrained(base, repo_id_or_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            repo_id_or_path, num_labels=3
        )

    model.to(device)
    # Inference mode (equivalent to PyTorch's shorter-named method; we use this
    # spelling consistently in the codebase — see Phase 3 supplement note).
    model.train(False)
    return model, tokenizer
