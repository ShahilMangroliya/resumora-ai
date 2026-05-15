import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from huggingface_hub import ModelCard

from training.publish import model as publish_model


def test_build_model_card_includes_required_disclosures():
    card = publish_model.build_model_card(
        base_model="distilbert-base-uncased",
        dataset_repo="alice/resumora-ai-dataset",
        train_config={"num_train_epochs": 3, "seed": 42, "max_length": 512},
        eval_metrics={
            "accuracy": 0.72,
            "macro_f1": 0.69,
            "per_class_f1": {"weak": 0.7, "partial": 0.6, "strong": 0.8},
            "mae": 12.4,
            "n": 30,
        },
    )
    assert isinstance(card, ModelCard)
    text = card.content
    # Disclosures the supplement §7.1 mandates.
    assert "score range" in text.lower() or "[20, 85]" in text or "20-85" in text
    assert "not for hiring decisions" in text.lower()
    assert "synthetic" in text.lower()
    assert "alice/resumora-ai-dataset" in text
    assert "ollama" in text.lower()
    assert "llama3.2:3b" in text.lower()
    assert "apache-2.0" in text.lower()
    # Metrics surfaced numerically.
    assert "0.72" in text or "72" in text


def test_push_model_calls_create_repo_and_upload_folder(tmp_path: Path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "adapter_config.json").write_text("{}")

    api = MagicMock()
    monkeypatch.setattr(publish_model, "HfApi", lambda token=None: api)

    card = MagicMock()
    card.push_to_hub = MagicMock()

    publish_model.push_model(
        repo_id="alice/resumora-ai-distilbert-lora",
        model_dir=model_dir,
        model_card=card,
        hf_token="hf_test_token",
        commit_message="phase 3 — initial release",
    )

    api.create_repo.assert_called_once()
    create_kwargs = api.create_repo.call_args.kwargs
    assert create_kwargs["repo_id"] == "alice/resumora-ai-distilbert-lora"
    assert create_kwargs["repo_type"] == "model"
    assert create_kwargs["exist_ok"] is True

    api.upload_folder.assert_called_once()
    upload_kwargs = api.upload_folder.call_args.kwargs
    assert upload_kwargs["repo_id"] == "alice/resumora-ai-distilbert-lora"
    assert upload_kwargs["repo_type"] == "model"
    assert upload_kwargs["folder_path"] == str(model_dir)
    assert upload_kwargs["commit_message"] == "phase 3 — initial release"

    card.push_to_hub.assert_called_once_with(
        "alice/resumora-ai-distilbert-lora", token="hf_test_token"
    )


def test_push_model_refuses_blank_token(tmp_path: Path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setattr(publish_model, "HfApi", lambda token=None: MagicMock())
    with pytest.raises(ValueError, match="HF token"):
        publish_model.push_model(
            repo_id="alice/m",
            model_dir=model_dir,
            model_card=MagicMock(),
            hf_token="",
        )


def test_push_model_refuses_missing_model_dir(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(publish_model, "HfApi", lambda token=None: MagicMock())
    with pytest.raises(FileNotFoundError):
        publish_model.push_model(
            repo_id="alice/m",
            model_dir=tmp_path / "nope",
            model_card=MagicMock(),
            hf_token="hf_test_token",
        )


def test_main_threads_args_through(tmp_path: Path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    metrics = {
        "accuracy": 0.7, "macro_f1": 0.65,
        "per_class_f1": {"weak": 0.6, "partial": 0.6, "strong": 0.7},
        "mae": 14.0, "n": 30,
    }
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics))

    called: dict = {}

    def fake_push(repo_id, model_dir, model_card, hf_token, commit_message="update model"):
        called["repo_id"] = repo_id
        called["model_dir"] = model_dir
        called["hf_token"] = hf_token

    monkeypatch.setattr(publish_model, "push_model", fake_push)
    monkeypatch.setenv("HF_TOKEN", "hf_env_token")

    publish_model.main([
        "--repo", "alice/m",
        "--model-dir", str(model_dir),
        "--metrics-json", str(metrics_path),
        "--dataset-repo", "alice/d",
        "--base-model", "distilbert-base-uncased",
        "--train-config-json", "{}",
    ])

    assert called["repo_id"] == "alice/m"
    assert called["model_dir"] == model_dir
    assert called["hf_token"] == "hf_env_token"
