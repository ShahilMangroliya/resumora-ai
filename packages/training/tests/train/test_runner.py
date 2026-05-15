from pathlib import Path

import pytest
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import DistilBertConfig, DistilBertForSequenceClassification

from training.train import runner as runner_mod


def _wrapped_tiny() -> torch.nn.Module:
    cfg = DistilBertConfig(
        vocab_size=200,
        max_position_embeddings=64,
        dim=32,
        n_layers=2,
        n_heads=2,
        hidden_dim=64,
        num_labels=3,
    )
    base = DistilBertForSequenceClassification(cfg)
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_lin", "v_lin"],
        modules_to_save=["pre_classifier", "classifier"],
        bias="none",
    )
    return get_peft_model(base, lora_cfg)


def _sample_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
        "attention_mask": torch.ones((2, 4), dtype=torch.long),
        "labels": torch.tensor([0, 2]),
    }


def test_verify_head_receives_gradients_passes_for_correct_config():
    model = _wrapped_tiny()
    runner_mod.verify_head_receives_gradients(model, _sample_batch())  # must not raise


def test_verify_head_receives_gradients_raises_when_pre_classifier_is_frozen():
    """If modules_to_save misses pre_classifier, the guard must fail loud."""
    model = _wrapped_tiny()
    # Simulate the bug: freeze pre_classifier so its gradients are not computed.
    for name, p in model.named_parameters():
        if "pre_classifier" in name:
            p.requires_grad = False
    with pytest.raises(RuntimeError, match="pre_classifier"):
        runner_mod.verify_head_receives_gradients(model, _sample_batch())


def test_train_result_dataclass_fields_exist():
    result = runner_mod.TrainResult(
        output_dir=Path("/tmp/x"),
        mlflow_run_id="abc",
        final_metrics={"accuracy": 1.0},
    )
    assert result.output_dir == Path("/tmp/x")
    assert result.mlflow_run_id == "abc"
    assert result.final_metrics["accuracy"] == 1.0


@pytest.mark.integration
def test_train_runs_end_to_end_on_a_handful_of_pairs(tmp_path: Path, monkeypatch):
    """Smoke: real DistilBERT, 6 pairs, 1 epoch -> adapter saved + metrics json written."""
    from training.dataset.jsonl import write_pairs
    from training.dataset.schema import Pair
    from training.train.config import TrainConfig

    def _pair(pid: str, label: str, score: int) -> Pair:
        return Pair(
            pair_id=pid,
            resume_text="alice has 5 years of python",
            jd_text="we need a python engineer",
            label=label,
            score=score,
            role="backend_dev",
            seniority="senior",
            source="synthetic",
            generator_model="llama3.2:3b",
            generated_at="2026-05-15T00:00:00Z",
            prompt_seed=0,
        )

    pairs_path = tmp_path / "pairs.jsonl"
    write_pairs(pairs_path, [
        _pair("p1", "weak", 20), _pair("p2", "weak", 20),
        _pair("p3", "partial", 55), _pair("p4", "partial", 55),
        _pair("p5", "strong", 85), _pair("p6", "strong", 85),
    ])

    cfg = TrainConfig(
        model_name="distilbert-base-uncased",
        train_pairs_path=pairs_path,
        gold_pairs_path=tmp_path / "gold.jsonl",  # not used during train
        output_dir=tmp_path / "out",
        max_length=32,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=5e-5,
        val_fraction=0.33,
        seed=42,
        mlflow_experiment="resumefit-test",
        run_name="integration-smoke",
    )

    # Keep MLflow inside tmp so the test does not pollute the repo's mlruns/.
    monkeypatch.chdir(tmp_path)

    result = runner_mod.train(cfg)
    assert (result.output_dir / "final_metrics.json").exists()
    assert isinstance(result.mlflow_run_id, str) and result.mlflow_run_id
