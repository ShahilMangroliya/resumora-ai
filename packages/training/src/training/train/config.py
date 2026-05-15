from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainConfig:
    """All inputs to a training run, in one place.

    Frozen so an in-flight run cannot mutate its own config — every mutation
    would invalidate the MLflow record.
    """

    model_name: str
    train_pairs_path: Path
    gold_pairs_path: Path
    output_dir: Path
    max_length: int
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    val_fraction: float
    seed: int
    mlflow_experiment: str
    run_name: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict — Paths become strings."""
        d = asdict(self)
        for key, value in d.items():
            if isinstance(value, Path):
                d[key] = str(value)
        return d


def smoke_config() -> TrainConfig:
    """Tiny config that runs on Mac CPU in under a minute.

    Used for smoke-testing the pipeline (no real learning happens).
    """
    return TrainConfig(
        model_name="distilbert-base-uncased",
        train_pairs_path=Path("data/synthetic/pairs.jsonl"),
        gold_pairs_path=Path("data/gold/seed.jsonl"),
        output_dir=Path("outputs/smoke"),
        max_length=128,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        val_fraction=0.2,
        seed=42,
        mlflow_experiment="resumora-ai",
        run_name="smoke",
    )


def full_config() -> TrainConfig:
    """Colab T4 config — the real training run."""
    return TrainConfig(
        model_name="distilbert-base-uncased",
        train_pairs_path=Path("data/synthetic/pairs.jsonl"),
        gold_pairs_path=Path("data/gold/seed.jsonl"),
        output_dir=Path("outputs/full"),
        max_length=512,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=5e-5,
        val_fraction=0.1,
        seed=42,
        mlflow_experiment="resumora-ai",
        run_name="full",
    )
