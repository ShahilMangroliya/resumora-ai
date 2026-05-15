from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from training.dataset.schema import Pair
from training.train.data import LABEL_TO_INT
from training.train.metrics import INT_TO_SCORE, score_from_logits


@dataclass
class EvalReport:
    """Structured eval result; JSON-serializable."""

    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    mae: float
    confusion_matrix: list[list[int]]
    n: int

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_against_gold(
    *,
    model,
    tokenizer,
    gold_pairs: list[Pair],
    max_length: int,
    device: str,
) -> EvalReport:
    """Predict on every gold pair and return aggregate metrics + confusion matrix."""
    # Switch to inference mode (disable dropout, freeze BatchNorm running stats).
    # Equivalent to model's shorter-named inference-mode method.
    model.train(False)
    logits_rows: list[list[float]] = []
    labels: list[int] = []

    for pair in gold_pairs:
        enc = tokenizer(
            pair.resume_text,
            pair.jd_text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            padding=True,
        )
        input_ids = enc["input_ids"].to(device) if hasattr(enc["input_ids"], "to") else enc["input_ids"]
        attention_mask = enc["attention_mask"].to(device) if hasattr(enc["attention_mask"], "to") else enc["attention_mask"]
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits_rows.append(out.logits[0].tolist())
        labels.append(LABEL_TO_INT[pair.label])

    logits = np.array(logits_rows)
    labels_arr = np.array(labels)
    preds = logits.argmax(axis=-1)

    accuracy = float((preds == labels_arr).mean()) if len(labels) else 0.0
    macro_f1 = float(f1_score(labels_arr, preds, average="macro", labels=[0, 1, 2], zero_division=0))
    per_class = f1_score(labels_arr, preds, average=None, labels=[0, 1, 2], zero_division=0)
    pred_scores = score_from_logits(logits)
    true_scores = np.array([INT_TO_SCORE[int(label)] for label in labels_arr])
    mae = float(np.abs(pred_scores - true_scores).mean()) if len(labels) else 0.0
    cm = confusion_matrix(labels_arr, preds, labels=[0, 1, 2]).tolist()

    return EvalReport(
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class_f1={
            "weak": float(per_class[0]),
            "partial": float(per_class[1]),
            "strong": float(per_class[2]),
        },
        mae=mae,
        confusion_matrix=cm,
        n=len(labels),
    )


def render_report(report: EvalReport) -> str:
    """Human-readable rendering for the CLI / notebook output."""
    lines = [
        f"n = {report.n}",
        f"accuracy = {report.accuracy:.3f}",
        f"macro_f1 = {report.macro_f1:.3f}",
        "per_class_f1:",
        f"  weak    = {report.per_class_f1['weak']:.3f}",
        f"  partial = {report.per_class_f1['partial']:.3f}",
        f"  strong  = {report.per_class_f1['strong']:.3f}",
        f"mae (score) = {report.mae:.2f}",
        "confusion (rows=true, cols=pred; order: weak/partial/strong):",
    ]
    for row in report.confusion_matrix:
        lines.append("  " + "  ".join(f"{v:>4d}" for v in row))
    return "\n".join(lines)
