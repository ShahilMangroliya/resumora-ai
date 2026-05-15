from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlflow
import torch
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from training.train.config import TrainConfig
from training.train.data import (
    assert_label_balance,
    build_dataset,
    load_pairs_local,
    train_val_split,
)
from training.train.metrics import compute_metrics
from training.train.model import build_model

_HEAD_GRAD_MIN_NORM = 1e-8  # any non-trivial signal is fine; "exactly zero" is the failure


@dataclass
class TrainResult:
    """Final state from a successful `train()` call."""

    output_dir: Path
    mlflow_run_id: str
    final_metrics: dict[str, float]


def verify_head_receives_gradients(
    model: torch.nn.Module,
    sample_batch: dict[str, torch.Tensor],
) -> None:
    """Run one forward+backward and raise if either head got no gradient.

    Catches the PEFT footgun where `modules_to_save` is missing `pre_classifier`
    (see supplement §3.1). Without this guard, training would silently
    underperform because half the head stays at init.
    """
    model.train()
    model.zero_grad(set_to_none=True)
    outputs = model(**sample_batch)
    outputs.loss.backward()

    for head_name in ("pre_classifier", "classifier"):
        grads = [
            p.grad for n, p in model.named_parameters()
            if head_name in n and p.grad is not None
        ]
        if not grads:
            raise RuntimeError(
                f"head {head_name!r} received no gradient — most likely missing "
                "from LoraConfig.modules_to_save"
            )
        total_norm = sum(g.norm().item() for g in grads)
        if total_norm < _HEAD_GRAD_MIN_NORM:
            raise RuntimeError(
                f"head {head_name!r} gradient norm is {total_norm:.2e} — head is "
                "effectively frozen"
            )

    model.zero_grad(set_to_none=True)


def train(config: TrainConfig) -> TrainResult:
    """Run a full training cycle and return the artifact location + run id."""
    pairs = load_pairs_local(config.train_pairs_path)
    assert_label_balance(pairs)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    full_ds = build_dataset(pairs, tokenizer=tokenizer, max_length=config.max_length)
    split = train_val_split(full_ds, val_fraction=config.val_fraction, seed=config.seed)

    model = build_model(config)

    # Guard the PEFT head-modules footgun before spinning up the Trainer.
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    sample = collator([split["train"][i] for i in range(min(2, len(split["train"])))])
    sample["labels"] = sample["labels"].long()
    verify_head_receives_gradients(model, sample)

    args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=config.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        seed=config.seed,
        report_to=["mlflow"],
        run_name=config.run_name,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["validation"],
        processing_class=tokenizer,  # transformers v5: replaces deprecated tokenizer= kwarg
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    mlflow.set_experiment(config.mlflow_experiment)
    with mlflow.start_run(run_name=config.run_name) as run:
        mlflow.log_params(_flatten_for_mlflow(config.to_dict()))
        trainer.train()
        final_metrics = trainer.evaluate()
        model.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
        (config.output_dir / "final_metrics.json").write_text(
            json.dumps(final_metrics, indent=2)
        )
        return TrainResult(
            output_dir=config.output_dir,
            mlflow_run_id=run.info.run_id,
            final_metrics=dict(final_metrics),
        )


def _flatten_for_mlflow(d: dict) -> dict:
    """MLflow params must be JSON-scalar; coerce non-scalars to strings."""
    return {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in d.items()}
