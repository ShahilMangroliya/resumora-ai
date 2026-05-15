"""Resumora AI fine-tuning entry points.

See docs/superpowers/specs/2026-05-15-phase-3-finetune-supplement.md for the
design.
"""

from training.train.config import TrainConfig, full_config, smoke_config
from training.train.evaluate import EvalReport, evaluate_against_gold, render_report
from training.train.runner import TrainResult, train, verify_head_receives_gradients

__all__ = [
    "EvalReport",
    "TrainConfig",
    "TrainResult",
    "evaluate_against_gold",
    "full_config",
    "render_report",
    "smoke_config",
    "train",
    "verify_head_receives_gradients",
]
