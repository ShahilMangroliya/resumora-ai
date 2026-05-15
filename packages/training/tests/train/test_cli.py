from pathlib import Path
from unittest.mock import MagicMock

import pytest

from training.train import cli


def test_train_subcommand_passes_preset_to_runner(monkeypatch, tmp_path: Path):
    captured: dict = {}

    def fake_train(config):
        captured["config"] = config
        return MagicMock(output_dir=tmp_path / "out", mlflow_run_id="abc", final_metrics={})

    monkeypatch.setattr(cli, "_train", fake_train)
    cli.main(["train", "--preset", "smoke"])
    assert captured["config"].run_name == "smoke"


def test_train_subcommand_overrides_flags_take_precedence(monkeypatch, tmp_path: Path):
    captured: dict = {}

    def fake_train(config):
        captured["config"] = config
        return MagicMock(output_dir=config.output_dir, mlflow_run_id="abc", final_metrics={})

    monkeypatch.setattr(cli, "_train", fake_train)
    cli.main([
        "train", "--preset", "smoke",
        "--train-pairs", str(tmp_path / "pairs.jsonl"),
        "--output-dir", str(tmp_path / "out"),
        "--num-epochs", "2",
    ])
    cfg = captured["config"]
    assert cfg.train_pairs_path == tmp_path / "pairs.jsonl"
    assert cfg.output_dir == tmp_path / "out"
    assert cfg.num_train_epochs == 2


def test_evaluate_subcommand_calls_runner(monkeypatch, tmp_path: Path):
    captured: dict = {}

    def fake_run_evaluate(model_dir, gold_pairs_path, max_length):
        captured["model_dir"] = model_dir
        captured["gold_pairs_path"] = gold_pairs_path
        captured["max_length"] = max_length
        return MagicMock()

    monkeypatch.setattr(cli, "_run_evaluate", fake_run_evaluate)
    cli.main([
        "evaluate",
        "--model-dir", str(tmp_path / "out"),
        "--gold-pairs", str(tmp_path / "gold.jsonl"),
        "--max-length", "256",
    ])
    assert captured["model_dir"] == tmp_path / "out"
    assert captured["gold_pairs_path"] == tmp_path / "gold.jsonl"
    assert captured["max_length"] == 256


def test_unknown_preset_errors():
    with pytest.raises(SystemExit):
        cli.main(["train", "--preset", "bogus"])
