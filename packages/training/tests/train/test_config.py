from pathlib import Path

import pytest

from training.train.config import TrainConfig, smoke_config, full_config


def test_train_config_is_frozen():
    cfg = TrainConfig(
        model_name="distilbert-base-uncased",
        train_pairs_path=Path("data/synthetic/pairs.jsonl"),
        gold_pairs_path=Path("data/gold/seed.jsonl"),
        output_dir=Path("outputs/run"),
        max_length=512,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=5e-5,
        val_fraction=0.1,
        seed=42,
        mlflow_experiment="resumefit",
        run_name="test",
    )
    with pytest.raises(Exception):
        cfg.model_name = "other"  # frozen


def test_smoke_config_is_cpu_friendly():
    cfg = smoke_config()
    assert cfg.num_train_epochs == 1
    assert cfg.per_device_train_batch_size <= 4
    assert cfg.max_length <= 128
    assert cfg.run_name.startswith("smoke")


def test_full_config_is_colab_sized():
    cfg = full_config()
    assert cfg.num_train_epochs >= 3
    assert cfg.per_device_train_batch_size >= 8
    assert cfg.max_length == 512
    assert cfg.run_name.startswith("full")


def test_to_dict_roundtrips():
    cfg = smoke_config()
    d = cfg.to_dict()
    # Paths are stringified so the dict is JSON-serializable for MLflow.
    assert isinstance(d["train_pairs_path"], str)
    assert d["num_train_epochs"] == cfg.num_train_epochs
    assert d["model_name"] == cfg.model_name
