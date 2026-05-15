from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from training.train.config import TrainConfig, full_config, smoke_config

_PRESETS = {"smoke": smoke_config, "full": full_config}


def _train(config: TrainConfig):
    """Lazy import — keeps `--help` snappy and avoids pulling torch for argparse tests."""
    from training.train.runner import train as run_train

    return run_train(config)


def _run_evaluate(model_dir: Path, gold_pairs_path: Path, max_length: int):
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from training.dataset.jsonl import read_pairs
    from training.train.evaluate import evaluate_against_gold, render_report

    # The training runner saves a PEFT adapter, not a full model. Loading
    # AutoModelForSequenceClassification on model_dir directly would silently
    # rebuild the base head from random init. Pull the base from the adapter
    # config and reattach the trained adapter on top.
    peft_cfg = PeftConfig.from_pretrained(model_dir)
    base = AutoModelForSequenceClassification.from_pretrained(
        peft_cfg.base_model_name_or_path, num_labels=3,
    )
    model = PeftModel.from_pretrained(base, model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    gold = read_pairs(gold_pairs_path)
    report = evaluate_against_gold(
        model=model, tokenizer=tokenizer, gold_pairs=gold, max_length=max_length, device="cpu",
    )
    print(render_report(report))
    return report


def _build_config_from_args(args: argparse.Namespace) -> TrainConfig:
    base = _PRESETS[args.preset]()
    overrides: dict = {}
    if args.train_pairs is not None:
        overrides["train_pairs_path"] = args.train_pairs
    if args.gold_pairs is not None:
        overrides["gold_pairs_path"] = args.gold_pairs
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.num_epochs is not None:
        overrides["num_train_epochs"] = args.num_epochs
    if args.batch_size is not None:
        overrides["per_device_train_batch_size"] = args.batch_size
    if args.run_name is not None:
        overrides["run_name"] = args.run_name
    return replace(base, **overrides)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="training.train")
    sub = parser.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="run a training cycle")
    tr.add_argument("--preset", choices=list(_PRESETS.keys()), default="smoke")
    tr.add_argument("--train-pairs", type=Path)
    tr.add_argument("--gold-pairs", type=Path)
    tr.add_argument("--output-dir", type=Path)
    tr.add_argument("--num-epochs", type=int)
    tr.add_argument("--batch-size", type=int)
    tr.add_argument("--run-name", type=str)

    ev = sub.add_parser("evaluate", help="evaluate a saved model against a gold set")
    ev.add_argument("--model-dir", type=Path, required=True)
    ev.add_argument("--gold-pairs", type=Path, required=True)
    ev.add_argument("--max-length", type=int, default=512)

    args = parser.parse_args(argv)

    if args.cmd == "train":
        cfg = _build_config_from_args(args)
        result = _train(cfg)
        print(f"run_id={result.mlflow_run_id} output_dir={result.output_dir}")
        print(json.dumps(result.final_metrics, indent=2))
    elif args.cmd == "evaluate":
        _run_evaluate(args.model_dir, args.gold_pairs, args.max_length)


if __name__ == "__main__":
    main()
