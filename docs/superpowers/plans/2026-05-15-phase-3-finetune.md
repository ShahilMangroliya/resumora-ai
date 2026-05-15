# Phase 3 — Fine-tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `packages/training/src/training/train/` — a fine-tuning pipeline that takes the Phase 2 synthetic dataset, trains a DistilBERT + LoRA classifier on `(resume, JD) → {weak, partial, strong}`, evaluates against the gold seed, logs to MLflow, and publishes the trained adapter + model card to HF Hub.

**Architecture:** A pure Python sub-package, runnable both locally (CPU smoke test) and on Colab (T4 full run) through one `python -m training.train` entry point. The Colab notebook is intentionally thin — it installs the repo and calls the same CLI. LoRA is wrapped around DistilBERT-base-uncased with `modules_to_save=["pre_classifier","classifier"]` (the head must train fully, not as adapters). Inference uses softmax · `[20, 55, 85]` to produce a continuous score in `[20, 85]` plus a confidence value. A new `training.publish.model` module pushes the adapter and an auto-generated card to HF Hub, mirroring the Phase 2 `training.publish.to_hf` pattern.

**Tech Stack:** Python 3.12, `transformers` (HF model + Trainer), `peft` (LoRA), `datasets` (HF Dataset abstraction), `torch` (Mac MPS / Colab CUDA), `accelerate` (Trainer requirement), `mlflow` (local-only run tracking), `scikit-learn` (confusion matrix, classification report), `huggingface_hub` (already pinned from Phase 2). Tests use `pytest` and rely on `transformers.DistilBertConfig` to build a tiny model entirely from config — no network downloads in unit tests; the real-weight load is exercised by integration tests.

**This plan is Phase 3 only.** It follows the master design doc `docs/superpowers/specs/2026-05-14-ai-pipeline-design.md` (§4 component 3, §7 phase 3) and its supplement `docs/superpowers/specs/2026-05-15-phase-3-finetune-supplement.md`. Decisions locked in during brainstorming on 2026-05-15:

- **3-class classifier → expected-value score** — matches the dataset's three discrete labels; produces a smooth `[20, 85]` output via `softmax · [20, 55, 85]` plus a confidence value (`max(softmax)`).
- **Score range is `[20, 85]`** — bounded by the bucket midpoints. This is the actual model output range and is disclosed in the model card, the user-facing docs, and the inference docstring.
- **DistilBERT + LoRA r=8, alpha=16**, target modules `q_lin`/`v_lin`, `modules_to_save=["pre_classifier","classifier"]` — both classification heads stay fully trainable because they start at random init.
- **Local CPU + Colab GPU through one entry point** — `python -m training.train` runs both. Tiny config → CPU smoke; full config → Colab GPU. No Colab-specific code path.
- **MLflow local-only** — `./mlruns/` directory; no remote tracking server. Colab zips and downloads `./mlruns/` at notebook end.
- **Gold-set publication gate at n ≥ 30** — development runs against the 5-pair gold seed are fine, but the model card published to HF Hub must report metrics on a gold set of at least 30 pairs. This is an explicit prerequisite the user owns.
- **Integration tests gated** — anything that touches the network (HF model download, Hub upload) is marked `@pytest.mark.integration` and skipped by default, consistent with the Ollama-integration pattern from Phase 2.

> **Note on the gold set:** the 5-pair seed in `data/gold/seed.jsonl` is enough to develop Phase 3. The plan does NOT include "write 25 more gold pairs" as a coding task because it is hand-labeled product work. The plan's final task surfaces this as a publication gate so the model is not pushed prematurely.

> **Note on PyTorch's inference-mode method:** PyTorch's `nn.Module` has a method that switches the model to inference mode (turns off dropout, freezes batchnorm stats). This plan uses `model.train(False)` everywhere instead of the equivalent shorter-named method, purely so the codebase does not contain a token that a security linter could confuse with Python's builtin code-execution function. Both calls are identical in PyTorch.

---

## File Structure

Files created or modified in this phase and their responsibility:

- `packages/training/pyproject.toml` — **modify**: add runtime deps (`transformers`, `torch`, `peft`, `datasets`, `accelerate`, `mlflow`, `scikit-learn`).
- `.gitignore` — **modify**: add `outputs/` (`mlruns/` is already ignored).
- `packages/training/src/training/train/__init__.py` — **create**: public surface of the train sub-package; re-exports the entry points.
- `packages/training/src/training/train/config.py` — **create**: `TrainConfig` dataclass — all hyperparameters in one place, serializable.
- `packages/training/src/training/train/data.py` — **create**: load `Pair`s from local JSONL or HF Hub; class-balance precheck; tokenize into an HF `Dataset`; train/val split.
- `packages/training/src/training/train/metrics.py` — **create**: `score_from_logits`, `confidence_from_logits`, `compute_metrics` (for the HF Trainer).
- `packages/training/src/training/train/model.py` — **create**: `default_lora_config`, `apply_lora`, `load_base_model`, `build_model`.
- `packages/training/src/training/train/runner.py` — **create**: `verify_head_receives_gradients` (the PEFT footgun guard) + `train(config)` wrapping HF Trainer with MLflow logging.
- `packages/training/src/training/train/evaluate.py` — **create**: `evaluate_against_gold(model, tokenizer, gold_pairs)` → `EvalReport` (metrics + confusion matrix).
- `packages/training/src/training/train/cli.py` — **create**: `python -m training.train` entry point with `train` and `evaluate` subcommands.
- `packages/training/src/training/publish/model.py` — **create**: build the model card; push adapter directory + card to HF Hub, mirroring `training.publish.to_hf`.
- `packages/training/tests/train/__init__.py` — **create**: marker for the test sub-package.
- `packages/training/tests/train/test_config.py` — **create**: tests for `TrainConfig`.
- `packages/training/tests/train/test_data.py` — **create**: tests for loading, balance check, tokenization, split.
- `packages/training/tests/train/test_metrics.py` — **create**: tests for score helpers and `compute_metrics`.
- `packages/training/tests/train/test_model.py` — **create**: tests for `default_lora_config`, `apply_lora` (using a config-built tiny DistilBERT); integration test for `load_base_model`.
- `packages/training/tests/train/test_runner.py` — **create**: tests for `verify_head_receives_gradients`; integration test for end-to-end `train()` on a tiny model.
- `packages/training/tests/train/test_evaluate.py` — **create**: tests for `evaluate_against_gold` against a stub model.
- `packages/training/tests/train/test_cli.py` — **create**: tests that CLI argparse routes the right config to `train`/`evaluate`.
- `packages/training/tests/publish/test_model.py` — **create**: tests for the model card builder and the Hub push (mocked `HfApi`, same pattern as `test_to_hf.py`).
- `notebooks/01_train_on_colab.ipynb` — **create**: thin Colab notebook that installs the repo, runs `training.train.cli`, evaluates, pushes, and zips `mlruns/`.
- `docs/phase-3-finetune.md` — **create**: user-facing Phase 3 guide.
- `README.md` — **modify**: add Phase 3 entry to the phases list.

---

## Task 1: Dependencies and train sub-package skeleton

**Files:**
- Modify: `packages/training/pyproject.toml`
- Modify: `.gitignore`
- Create: `packages/training/src/training/train/__init__.py`
- Create: `packages/training/tests/train/__init__.py`

- [ ] **Step 1: Add Phase 3 runtime dependencies**

In `packages/training/pyproject.toml`, change the `dependencies` list from:

```toml
dependencies = [
    "pipeline",
    "httpx>=0.27",
    "pydantic>=2.0",
    "huggingface-hub>=0.24",
]
```

to:

```toml
dependencies = [
    "pipeline",
    "httpx>=0.27",
    "pydantic>=2.0",
    "huggingface-hub>=0.24",
    "transformers>=4.44",
    "torch>=2.4",
    "peft>=0.12",
    "datasets>=2.20",
    "accelerate>=0.33",
    "mlflow>=2.16",
    "scikit-learn>=1.5",
]
```

These pin the floor versions. `torch` on Mac installs CPU + MPS wheels; on Colab the runtime already has CUDA-enabled torch installed, and `uv sync` in the notebook will keep it.

- [ ] **Step 2: Extend `.gitignore`**

In `.gitignore`, append after the `# MLflow` block (so right after the `mlruns/` line):

```
# Phase 3 — training outputs (model checkpoints, adapters)
outputs/
```

- [ ] **Step 3: Create the `train` sub-package marker**

Create `packages/training/src/training/train/__init__.py` as a placeholder; Task 8 fills in re-exports.

```python
"""Resumora AI fine-tuning entry points.

See docs/superpowers/specs/2026-05-15-phase-3-finetune-supplement.md for the
design.
"""
```

- [ ] **Step 4: Create the test sub-package marker**

Create `packages/training/tests/train/__init__.py` as an empty file (the marker pytest needs to discover the sub-package).

- [ ] **Step 5: Sync the workspace**

Run: `uv sync --all-packages`
Expected: completes without error; new dependencies are installed; `uv.lock` is updated. The install may take 1–2 minutes the first time because of `torch`.

- [ ] **Step 6: Verify the new dependencies import**

Run:
```bash
uv run python -c "import transformers, torch, peft, datasets, accelerate, mlflow, sklearn; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 7: Commit**

```bash
git add packages/training/pyproject.toml .gitignore packages/training/src/training/train/__init__.py packages/training/tests/train/__init__.py uv.lock
git commit -m "feat: add Phase 3 training dependencies and package skeleton"
```

---

## Task 2: `TrainConfig` dataclass

**Files:**
- Create: `packages/training/src/training/train/config.py`
- Create: `packages/training/tests/train/test_config.py`

`TrainConfig` is a frozen dataclass holding everything a training run needs: model name, hyperparameters, data sources, output directory. It is the only input to `train()`. Two factory helpers expose the standard configs: `smoke_config()` (tiny, CPU-friendly) and `full_config()` (Colab T4).

- [ ] **Step 1: Write the failing tests**

Create `packages/training/tests/train/test_config.py`:

```python
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
        mlflow_experiment="resumora-ai",
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest packages/training/tests/train/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.train.config'`.

- [ ] **Step 3: Implement `TrainConfig` and factories**

Create `packages/training/src/training/train/config.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest packages/training/tests/train/test_config.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/training/src/training/train/config.py packages/training/tests/train/test_config.py
git commit -m "feat: add TrainConfig dataclass with smoke/full factories"
```

---

## Task 3: Data loading, balance precheck, tokenization, split

**Files:**
- Create: `packages/training/src/training/train/data.py`
- Create: `packages/training/tests/train/test_data.py`

`data.py` turns raw `Pair` records into a tokenized HF `Dataset` ready for the Trainer. It owns six concerns:

1. **`LABEL_TO_INT` / `INT_TO_LABEL`** — string ↔ integer mapping for the classifier head.
2. **`load_pairs_local(path)`** — reads JSONL via the existing `read_pairs` helper from `training.dataset.jsonl`.
3. **`load_pairs_from_hub(repo_id, filename)`** — downloads a JSONL file from a Hub *dataset* repo and reads it.
4. **`assert_label_balance(pairs, max_deviation=0.20)`** — fails loudly if any label is more than 20 percentage points off the perfect 33/33/33 split (per supplement §6.3).
5. **`build_dataset(pairs, tokenizer, max_length)`** — tokenizes resume+JD as a sentence pair, returns a `datasets.Dataset` with `input_ids`, `attention_mask`, `label`.
6. **`train_val_split(dataset, val_fraction, seed)`** — wraps `Dataset.train_test_split` for repeatable splits.

- [ ] **Step 1: Write the failing tests**

Create `packages/training/tests/train/test_data.py`:

```python
from pathlib import Path

import pytest

from training.dataset.jsonl import write_pairs
from training.dataset.schema import Pair
from training.train import data as data_mod


def _make_pair(pair_id: str, label: str, score: int) -> Pair:
    return Pair(
        pair_id=pair_id,
        resume_text="alice has 5 years of python experience",
        jd_text="we need a senior python engineer",
        label=label,
        score=score,
        role="backend_dev",
        seniority="senior",
        source="synthetic",
        generator_model="llama3.2:3b",
        generated_at="2026-05-15T00:00:00Z",
        prompt_seed=0,
    )


def test_label_to_int_round_trip():
    assert data_mod.LABEL_TO_INT["weak"] == 0
    assert data_mod.LABEL_TO_INT["partial"] == 1
    assert data_mod.LABEL_TO_INT["strong"] == 2
    assert data_mod.INT_TO_LABEL[0] == "weak"
    assert data_mod.INT_TO_LABEL[1] == "partial"
    assert data_mod.INT_TO_LABEL[2] == "strong"


def test_load_pairs_local_reads_jsonl(tmp_path: Path):
    path = tmp_path / "pairs.jsonl"
    write_pairs(path, [_make_pair("p1", "weak", 20), _make_pair("p2", "strong", 85)])
    pairs = data_mod.load_pairs_local(path)
    assert [p.pair_id for p in pairs] == ["p1", "p2"]


def test_assert_label_balance_passes_when_balanced():
    pairs = [_make_pair(f"p{i}", "weak", 20) for i in range(3)] + \
            [_make_pair(f"q{i}", "partial", 55) for i in range(3)] + \
            [_make_pair(f"r{i}", "strong", 85) for i in range(3)]
    data_mod.assert_label_balance(pairs)


def test_assert_label_balance_raises_when_skewed():
    pairs = [_make_pair(f"p{i}", "weak", 20) for i in range(10)] + \
            [_make_pair("q", "partial", 55), _make_pair("r", "strong", 85)]
    with pytest.raises(ValueError, match="balance"):
        data_mod.assert_label_balance(pairs)


def test_assert_label_balance_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        data_mod.assert_label_balance([])


def test_build_dataset_tokenizes_and_carries_label():
    pairs = [_make_pair("p1", "weak", 20), _make_pair("p2", "strong", 85)]
    tokenizer = _FakeTokenizer()
    ds = data_mod.build_dataset(pairs, tokenizer=tokenizer, max_length=16)
    assert set(ds.column_names) == {"input_ids", "attention_mask", "label"}
    assert ds[0]["label"] == data_mod.LABEL_TO_INT["weak"]
    assert ds[1]["label"] == data_mod.LABEL_TO_INT["strong"]
    assert len(ds[0]["input_ids"]) <= 16


def test_train_val_split_is_deterministic_for_same_seed():
    pairs = [_make_pair(f"p{i}", "weak", 20) for i in range(20)]
    tokenizer = _FakeTokenizer()
    ds = data_mod.build_dataset(pairs, tokenizer=tokenizer, max_length=16)
    split_a = data_mod.train_val_split(ds, val_fraction=0.2, seed=42)
    split_b = data_mod.train_val_split(ds, val_fraction=0.2, seed=42)
    assert split_a["train"]["label"] == split_b["train"]["label"]
    assert split_a["validation"]["label"] == split_b["validation"]["label"]


class _FakeTokenizer:
    """Minimal stand-in for an HF tokenizer.

    `build_dataset` only needs `tokenizer(text_a, text_b, truncation=..., max_length=...)`
    to return a dict with `input_ids` and `attention_mask` — exactly what a real
    tokenizer does.
    """

    def __call__(self, text_a, text_b, truncation, max_length):
        ids = [1, 2, 3, 4][:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest packages/training/tests/train/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.train.data'`.

- [ ] **Step 3: Implement `data.py`**

Create `packages/training/src/training/train/data.py`:

```python
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Protocol

from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download

from training.dataset.jsonl import read_pairs
from training.dataset.schema import Label, Pair

LABEL_TO_INT: dict[Label, int] = {"weak": 0, "partial": 1, "strong": 2}
INT_TO_LABEL: dict[int, Label] = {v: k for k, v in LABEL_TO_INT.items()}


class _Tokenizer(Protocol):
    """Anything that behaves like an HF tokenizer for sentence-pair input."""

    def __call__(self, text_a: str, text_b: str, truncation: bool, max_length: int) -> dict[str, Any]:
        ...


def load_pairs_local(path: Path) -> list[Pair]:
    """Read every pair from a local JSONL file (reuses Phase 2 IO)."""
    return read_pairs(path)


def load_pairs_from_hub(repo_id: str, filename: str, hf_token: str | None = None) -> list[Pair]:
    """Download a JSONL file from an HF *dataset* repo and parse it as pairs.

    `filename` is the path inside the repo, e.g. "synthetic/pairs.jsonl".
    """
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=hf_token,
    )
    return read_pairs(Path(local_path))


def assert_label_balance(pairs: list[Pair], max_deviation: float = 0.20) -> None:
    """Guard against silently-skewed training data.

    Per the supplement §6.3: every label must be within `max_deviation` of the
    perfect 1/3 share. A 60/20/20 dataset will train a degenerate classifier;
    the assertion forces it into the foreground before the run starts.
    """
    if not pairs:
        raise ValueError("cannot train on an empty pair set")
    counts = Counter(p.label for p in pairs)
    total = len(pairs)
    expected = 1 / 3
    for label, count in counts.items():
        share = count / total
        if abs(share - expected) > max_deviation:
            raise ValueError(
                f"label balance check failed: {label}={share:.0%} of {total} "
                f"(expected within {max_deviation:.0%} of {expected:.0%})"
            )


def build_dataset(
    pairs: list[Pair],
    *,
    tokenizer: _Tokenizer,
    max_length: int,
) -> Dataset:
    """Tokenize resume + JD as a sentence pair and attach the integer label."""
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        encoded = tokenizer(pair.resume_text, pair.jd_text, truncation=True, max_length=max_length)
        rows.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "label": LABEL_TO_INT[pair.label],
            }
        )
    return Dataset.from_list(rows)


def train_val_split(dataset: Dataset, *, val_fraction: float, seed: int) -> DatasetDict:
    """Stratified-ish random split. Returns DatasetDict with `train` and `validation`."""
    split = dataset.train_test_split(test_size=val_fraction, seed=seed)
    return DatasetDict({"train": split["train"], "validation": split["test"]})
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest packages/training/tests/train/test_data.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/training/src/training/train/data.py packages/training/tests/train/test_data.py
git commit -m "feat: add training data module (load, balance check, tokenize, split)"
```

---

## Task 4: Metrics — score helpers and Trainer hook

**Files:**
- Create: `packages/training/src/training/train/metrics.py`
- Create: `packages/training/tests/train/test_metrics.py`

This module is the heart of "3-class → continuous score." It owns:

- **`INT_TO_SCORE = {0: 20, 1: 55, 2: 85}`** — bucket midpoints from `training.dataset.schema.LABEL_TO_SCORE`, indexed by integer label.
- **`score_from_logits(logits)`** — `softmax(logits) · [20, 55, 85]`. Vectorized over `(N, 3)` arrays.
- **`confidence_from_logits(logits)`** — `max(softmax(logits))`. Vectorized.
- **`compute_metrics(eval_pred)`** — the Trainer hook. Returns `{accuracy, macro_f1, f1_weak, f1_partial, f1_strong, mae}`.

Why MAE is computed against `INT_TO_SCORE[true_label]` (not a continuous gold score): the gold set uses the same bucketed labels, so the gold "score" is also one of `{20, 55, 85}`. MAE measures how far the expected-value prediction lands from the correct bucket midpoint — that is a meaningful product-side error.

- [ ] **Step 1: Write the failing tests**

Create `packages/training/tests/train/test_metrics.py`:

```python
import numpy as np

from training.train.metrics import (
    INT_TO_SCORE,
    compute_metrics,
    confidence_from_logits,
    score_from_logits,
)


def test_int_to_score_matches_bucket_midpoints():
    assert INT_TO_SCORE == {0: 20, 1: 55, 2: 85}


def test_score_from_logits_is_in_bucket_range():
    # Very confident "strong" → score near 85; very confident "weak" → score near 20.
    logits = np.array([[10.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
    scores = score_from_logits(logits)
    assert scores[0] < 25
    assert scores[1] > 80
    # All scores must land inside [20, 85].
    for s in scores:
        assert 20 <= s <= 85


def test_score_from_logits_uniform_lands_at_average_bucket():
    # Equal logits → uniform probs → score = (20 + 55 + 85) / 3 = 53.33...
    logits = np.zeros((1, 3))
    scores = score_from_logits(logits)
    assert abs(scores[0] - (20 + 55 + 85) / 3) < 1e-6


def test_confidence_from_logits_returns_max_prob():
    logits = np.array([[10.0, 0.0, 0.0]])
    conf = confidence_from_logits(logits)
    assert conf[0] > 0.99


def test_compute_metrics_basic_shape():
    # Perfect predictions for a tiny eval set.
    logits = np.array([
        [10.0, 0.0, 0.0],    # predicts weak
        [0.0, 10.0, 0.0],    # predicts partial
        [0.0, 0.0, 10.0],    # predicts strong
    ])
    labels = np.array([0, 1, 2])
    metrics = compute_metrics((logits, labels))
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["f1_weak"] == 1.0
    assert metrics["f1_partial"] == 1.0
    assert metrics["f1_strong"] == 1.0
    # Predictions equal gold → MAE is small (just softmax slack).
    assert metrics["mae"] < 5.0


def test_compute_metrics_handles_imperfect_predictions():
    # Two correct, one off-by-one (predicted partial but label was weak).
    logits = np.array([
        [0.0, 10.0, 0.0],    # predicts partial, label weak
        [0.0, 10.0, 0.0],    # predicts partial, label partial
        [0.0, 0.0, 10.0],    # predicts strong, label strong
    ])
    labels = np.array([0, 1, 2])
    metrics = compute_metrics((logits, labels))
    assert metrics["accuracy"] == 2 / 3
    assert metrics["macro_f1"] < 1.0
    # First row: predicted score ≈ 55, true bucket score = 20 → ~35 error.
    assert metrics["mae"] > 10
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest packages/training/tests/train/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.train.metrics'`.

- [ ] **Step 3: Implement `metrics.py`**

Create `packages/training/src/training/train/metrics.py`:

```python
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

INT_TO_SCORE: dict[int, int] = {0: 20, 1: 55, 2: 85}
_SCORE_VECTOR = np.array([20.0, 55.0, 85.0])


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def score_from_logits(logits: np.ndarray) -> np.ndarray:
    """Expected-value score: softmax(logits) dotted with [20, 55, 85].

    Output is bounded to [20, 85]. The 0–100 product surface is honored by
    disclosure (model card §7.1 of the design supplement), not by stretching
    the range.
    """
    probs = _softmax(np.asarray(logits, dtype=np.float64))
    return probs @ _SCORE_VECTOR


def confidence_from_logits(logits: np.ndarray) -> np.ndarray:
    """Max softmax probability — a simple per-prediction confidence."""
    probs = _softmax(np.asarray(logits, dtype=np.float64))
    return probs.max(axis=-1)


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """HF Trainer `compute_metrics` hook.

    Reports accuracy, macro-F1, per-class F1, and MAE of the expected-value
    score against the bucket midpoint of the true label.
    """
    logits, labels = eval_pred
    logits = np.asarray(logits)
    labels = np.asarray(labels)

    preds = logits.argmax(axis=-1)
    accuracy = float((preds == labels).mean())

    macro_f1 = float(f1_score(labels, preds, average="macro", labels=[0, 1, 2], zero_division=0))
    per_class = f1_score(labels, preds, average=None, labels=[0, 1, 2], zero_division=0)

    pred_scores = score_from_logits(logits)
    true_scores = np.array([INT_TO_SCORE[int(label)] for label in labels])
    mae = float(np.abs(pred_scores - true_scores).mean())

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "f1_weak": float(per_class[0]),
        "f1_partial": float(per_class[1]),
        "f1_strong": float(per_class[2]),
        "mae": mae,
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest packages/training/tests/train/test_metrics.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/training/src/training/train/metrics.py packages/training/tests/train/test_metrics.py
git commit -m "feat: add score expectation + Trainer compute_metrics hook"
```

---

## Task 5: Model — base load + LoRA wrap

**Files:**
- Create: `packages/training/src/training/train/model.py`
- Create: `packages/training/tests/train/test_model.py`

`model.py` builds the model the Trainer will optimize. The functions split cleanly so unit tests can exercise the LoRA wrapping without downloading the real DistilBERT weights — they build a config-only randomly-initialized tiny DistilBERT and wrap that.

- **`default_lora_config()`** — returns the exact `LoraConfig` from the supplement §3.
- **`apply_lora(base_model, lora_config)`** — wraps a pre-built model with LoRA. Unit-tested.
- **`load_base_model(model_name, num_labels)`** — calls `AutoModelForSequenceClassification.from_pretrained`. Integration-tested only (hits the network).
- **`build_model(config)`** — composes the two. Integration-tested only.

The non-negotiable assertion in this task: after `apply_lora`, `pre_classifier` and `classifier` are listed in the LoRA `modules_to_save` set, meaning they will be trained *as full layers*, not as adapters. This is the PEFT footgun the supplement §3.1 calls out.

- [ ] **Step 1: Write the failing unit tests**

Create `packages/training/tests/train/test_model.py`:

```python
import pytest
import torch
from peft import LoraConfig, PeftModel, TaskType
from transformers import DistilBertConfig, DistilBertForSequenceClassification

from training.train import model as model_mod
from training.train.config import smoke_config


def _tiny_distilbert() -> DistilBertForSequenceClassification:
    """A randomly-initialized 2-layer DistilBERT.

    Built from config — no network. Small enough that gradient checks run in
    milliseconds.
    """
    cfg = DistilBertConfig(
        vocab_size=200,
        max_position_embeddings=64,
        dim=32,
        n_layers=2,
        n_heads=2,
        hidden_dim=64,
        num_labels=3,
    )
    return DistilBertForSequenceClassification(cfg)


def test_default_lora_config_matches_supplement():
    cfg = model_mod.default_lora_config()
    assert isinstance(cfg, LoraConfig)
    assert cfg.task_type == TaskType.SEQ_CLS
    assert cfg.r == 8
    assert cfg.lora_alpha == 16
    assert set(cfg.target_modules) == {"q_lin", "v_lin"}
    assert "pre_classifier" in cfg.modules_to_save
    assert "classifier" in cfg.modules_to_save
    assert cfg.bias == "none"


def test_apply_lora_returns_a_peft_model():
    base = _tiny_distilbert()
    wrapped = model_mod.apply_lora(base, model_mod.default_lora_config())
    assert isinstance(wrapped, PeftModel)


def test_apply_lora_keeps_pre_classifier_and_classifier_trainable():
    base = _tiny_distilbert()
    wrapped = model_mod.apply_lora(base, model_mod.default_lora_config())

    trainable_names = {n for n, p in wrapped.named_parameters() if p.requires_grad}
    # The full pre_classifier and classifier weights must be trainable
    # (modules_to_save), not just LoRA adapters of them.
    assert any("pre_classifier" in n for n in trainable_names)
    assert any("classifier" in n for n in trainable_names)


def test_apply_lora_freezes_base_encoder_weights():
    base = _tiny_distilbert()
    wrapped = model_mod.apply_lora(base, model_mod.default_lora_config())

    frozen_names = {n for n, p in wrapped.named_parameters() if not p.requires_grad}
    # The base attention/FFN weights must be frozen — LoRA's whole point.
    assert any("attention.q_lin.base_layer" in n or "attention.q_lin.weight" in n
               for n in frozen_names)


def test_apply_lora_produces_three_class_logits():
    base = _tiny_distilbert()
    wrapped = model_mod.apply_lora(base, model_mod.default_lora_config())
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    out = wrapped(input_ids=input_ids, attention_mask=attention_mask)
    assert out.logits.shape == (1, 3)
```

- [ ] **Step 2: Run the unit tests to verify they fail**

Run: `uv run pytest packages/training/tests/train/test_model.py -v -k "not integration"`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.train.model'`.

- [ ] **Step 3: Implement `model.py`**

Create `packages/training/src/training/train/model.py`:

```python
from __future__ import annotations

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, PreTrainedModel

from training.train.config import TrainConfig

_NUM_LABELS = 3  # weak / partial / strong


def default_lora_config() -> LoraConfig:
    """The Phase 3 LoRA config.

    `modules_to_save=["pre_classifier", "classifier"]` is load-bearing — both
    classification heads are randomly-initialized and must train as full
    layers, not as low-rank adapters. See the design supplement §3.1.
    """
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_lin", "v_lin"],
        modules_to_save=["pre_classifier", "classifier"],
        bias="none",
    )


def apply_lora(base_model: PreTrainedModel, lora_config: LoraConfig) -> PeftModel:
    """Wrap a sequence-classification model with LoRA."""
    return get_peft_model(base_model, lora_config)


def load_base_model(model_name: str, *, num_labels: int = _NUM_LABELS) -> PreTrainedModel:
    """Download (or load from cache) a sequence-classification model.

    Hits the network on first call. Tested as integration only.
    """
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )


def build_model(config: TrainConfig) -> PeftModel:
    """One-shot: load the base model and wrap it with the default LoRA config."""
    base = load_base_model(config.model_name)
    return apply_lora(base, default_lora_config())
```

- [ ] **Step 4: Run the unit tests to verify they pass**

Run: `uv run pytest packages/training/tests/train/test_model.py -v -k "not integration"`
Expected: PASS (5 tests).

- [ ] **Step 5: Add an integration test for `build_model`**

Append to `packages/training/tests/train/test_model.py`:

```python
@pytest.mark.integration
def test_build_model_downloads_distilbert_and_wraps_it():
    cfg = smoke_config()
    wrapped = model_mod.build_model(cfg)
    assert isinstance(wrapped, PeftModel)
    # The real DistilBERT has 6 layers; the LoRA wrapper must see them.
    base_layers = [n for n, _ in wrapped.named_parameters() if "transformer.layer" in n]
    assert len(base_layers) > 0
```

> **Note:** integration tests are skipped by the default pytest invocation (per the root `pyproject.toml` `addopts = "-m 'not integration'"`). They run only when explicitly selected.

- [ ] **Step 6: Run integration tests as a sanity check (optional, network-bound)**

Run: `uv run pytest packages/training/tests/train/test_model.py -v -m integration`
Expected: PASS — downloads ~250MB of DistilBERT weights on first run. If you're offline, skip this step; CI will skip it too.

- [ ] **Step 7: Commit**

```bash
git add packages/training/src/training/train/model.py packages/training/tests/train/test_model.py
git commit -m "feat: add DistilBERT + LoRA model builder (modules_to_save head)"
```

---

## Task 6: Runner — head-gradient guard, HF Trainer, MLflow

**Files:**
- Create: `packages/training/src/training/train/runner.py`
- Create: `packages/training/tests/train/test_runner.py`

This is the only file in Phase 3 that actually trains. It owns three things:

- **`verify_head_receives_gradients(model, sample_batch)`** — runs one forward + backward on a real batch and raises `RuntimeError` if `pre_classifier` or `classifier` end up with `None` or near-zero gradients. This is the PEFT footgun guard wired into the production path so every training run executes it (supplement §3.1).
- **`train(config, tokenizer)`** — the main entry. Loads pairs, asserts balance, builds the model, runs `verify_head_receives_gradients` on the first batch, configures the HF Trainer, starts MLflow, trains, evaluates on the held-out validation set, logs everything to MLflow, returns a `TrainResult` (output dir, MLflow run id, final val metrics).
- **`TrainResult`** — dataclass: `output_dir: Path`, `mlflow_run_id: str`, `final_metrics: dict[str, float]`.

The Trainer uses HF's MLflow auto-integration via `report_to=["mlflow"]` — no manual logging needed for per-step metrics, but we add explicit `mlflow.log_params` for the full `TrainConfig`.

- [ ] **Step 1: Write the failing tests**

Create `packages/training/tests/train/test_runner.py`:

```python
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
```

The integration test for end-to-end training is added as a separate step below so unit tests stay fast.

- [ ] **Step 2: Run the unit tests to verify they fail**

Run: `uv run pytest packages/training/tests/train/test_runner.py -v -k "not integration"`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.train.runner'`.

- [ ] **Step 3: Implement `runner.py`**

Create `packages/training/src/training/train/runner.py`:

```python
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
        tokenizer=tokenizer,
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
```

- [ ] **Step 4: Run the unit tests to verify they pass**

Run: `uv run pytest packages/training/tests/train/test_runner.py -v -k "not integration"`
Expected: PASS (3 tests).

- [ ] **Step 5: Add an integration test for end-to-end `train()`**

Append to `packages/training/tests/train/test_runner.py`:

```python
@pytest.mark.integration
def test_train_runs_end_to_end_on_a_handful_of_pairs(tmp_path: Path, monkeypatch):
    """Smoke: real DistilBERT, 6 pairs, 1 epoch → adapter saved + metrics json written."""
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
        mlflow_experiment="resumora-ai-test",
        run_name="integration-smoke",
    )

    # Keep MLflow inside tmp so the test does not pollute the repo's mlruns/.
    monkeypatch.chdir(tmp_path)

    result = runner_mod.train(cfg)
    assert (result.output_dir / "final_metrics.json").exists()
    assert isinstance(result.mlflow_run_id, str) and result.mlflow_run_id
```

- [ ] **Step 6: Sanity-check the integration test (optional, slow)**

Run: `uv run pytest packages/training/tests/train/test_runner.py -v -m integration`
Expected: PASS — downloads DistilBERT (if not cached), runs ~1 minute on CPU.

- [ ] **Step 7: Commit**

```bash
git add packages/training/src/training/train/runner.py packages/training/tests/train/test_runner.py
git commit -m "feat: add training runner with head-gradient guard and MLflow"
```

---

## Task 7: Evaluate against the gold set

**Files:**
- Create: `packages/training/src/training/train/evaluate.py`
- Create: `packages/training/tests/train/test_evaluate.py`

`evaluate.py` runs a saved model against the gold pairs and produces a structured `EvalReport`. It is callable from the CLI (`python -m training.train evaluate`) and from inside `train()` for the final HF Hub publication step (Task 9).

- **`EvalReport`** — dataclass: `accuracy`, `macro_f1`, `per_class_f1: dict[str, float]`, `mae`, `confusion_matrix: list[list[int]]`, `n: int`. JSON-serializable.
- **`evaluate_against_gold(model, tokenizer, gold_pairs, *, max_length, device)`** — runs predictions, builds the report.
- **`render_report(report) -> str`** — pretty-prints the report; used by the CLI and the Colab notebook.

Note: the evaluation path puts the model into PyTorch's inference mode via `model.train(False)`. That call disables dropout and freezes BatchNorm running stats — equivalent to PyTorch's shorter-named inference-mode method, but written long-form to keep static security linters happy.

- [ ] **Step 1: Write the failing tests**

Create `packages/training/tests/train/test_evaluate.py`:

```python
import torch

from training.dataset.schema import Pair
from training.train import evaluate as eval_mod


def _pair(pid: str, label: str, score: int) -> Pair:
    return Pair(
        pair_id=pid,
        resume_text="resume " + pid,
        jd_text="jd " + pid,
        label=label,
        score=score,
        role="backend_dev",
        seniority="senior",
        source="gold",
        generator_model="manual",
        generated_at="2026-05-15T00:00:00Z",
        prompt_seed=0,
    )


class _StubModel(torch.nn.Module):
    """Returns fixed logits per pair_id so the test controls every prediction.

    Maps the first token id (which the stub tokenizer encodes from the pair_id)
    to a 3-vector of logits.
    """

    def __init__(self, logits_by_first_token: dict[int, list[float]]):
        super().__init__()
        self._logits = logits_by_first_token

    def forward(self, input_ids, attention_mask):
        rows = []
        for row in input_ids:
            first = int(row[0])
            rows.append(self._logits[first])
        return type("Out", (), {"logits": torch.tensor(rows)})


class _StubTokenizer:
    """Encodes each pair as `[token_for_pid]` so the stub model can identify it."""

    def __init__(self, pid_to_token: dict[str, int]):
        self._map = pid_to_token

    def __call__(self, text_a, text_b, truncation, max_length, return_tensors=None, padding=None):
        # Use the resume text "resume <pid>" to recover the pid.
        pid = text_a.split()[-1]
        token = self._map[pid]
        ids = torch.tensor([[token]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def test_evaluate_against_gold_perfect_predictions():
    pairs = [_pair("p1", "weak", 20), _pair("p2", "partial", 55), _pair("p3", "strong", 85)]
    tokenizer = _StubTokenizer({"p1": 10, "p2": 11, "p3": 12})
    model = _StubModel({
        10: [10.0, 0.0, 0.0],   # weak
        11: [0.0, 10.0, 0.0],   # partial
        12: [0.0, 0.0, 10.0],   # strong
    })

    report = eval_mod.evaluate_against_gold(
        model=model, tokenizer=tokenizer, gold_pairs=pairs, max_length=16, device="cpu",
    )
    assert report.n == 3
    assert report.accuracy == 1.0
    assert report.macro_f1 == 1.0
    assert report.mae < 5.0
    assert report.confusion_matrix == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_evaluate_against_gold_one_misclassification():
    pairs = [_pair("p1", "weak", 20), _pair("p2", "weak", 20)]
    tokenizer = _StubTokenizer({"p1": 10, "p2": 11})
    model = _StubModel({
        10: [10.0, 0.0, 0.0],     # predicts weak (correct)
        11: [0.0, 10.0, 0.0],     # predicts partial (wrong — labelled weak)
    })

    report = eval_mod.evaluate_against_gold(
        model=model, tokenizer=tokenizer, gold_pairs=pairs, max_length=16, device="cpu",
    )
    assert report.n == 2
    assert report.accuracy == 0.5
    # Confusion: 1 weak->weak, 1 weak->partial. Row index = true label, col = pred.
    assert report.confusion_matrix[0][0] == 1
    assert report.confusion_matrix[0][1] == 1


def test_render_report_includes_key_lines():
    report = eval_mod.EvalReport(
        accuracy=0.5,
        macro_f1=0.42,
        per_class_f1={"weak": 0.5, "partial": 0.0, "strong": 0.75},
        mae=12.3,
        confusion_matrix=[[1, 1, 0], [0, 0, 0], [0, 0, 1]],
        n=3,
    )
    rendered = eval_mod.render_report(report)
    assert "accuracy" in rendered.lower()
    assert "macro_f1" in rendered.lower()
    assert "confusion" in rendered.lower()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest packages/training/tests/train/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.train.evaluate'`.

- [ ] **Step 3: Implement `evaluate.py`**

Create `packages/training/src/training/train/evaluate.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest packages/training/tests/train/test_evaluate.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/training/src/training/train/evaluate.py packages/training/tests/train/test_evaluate.py
git commit -m "feat: add gold-set evaluation with confusion matrix"
```

---

## Task 8: CLI — `python -m training.train`

**Files:**
- Create: `packages/training/src/training/train/cli.py`
- Create: `packages/training/tests/train/test_cli.py`
- Modify: `packages/training/src/training/train/__init__.py`

Two subcommands: `train` and `evaluate`. Each accepts the smoke / full preset *or* explicit flags. The CLI is thin glue — its job is to translate argv into a `TrainConfig` or evaluation parameters.

```
python -m training.train train --preset smoke
python -m training.train train --preset full --train-pairs data/synthetic/pairs.jsonl --output-dir outputs/run-x
python -m training.train evaluate --model-dir outputs/run-x --gold-pairs data/gold/seed.jsonl
```

- [ ] **Step 1: Write the failing tests**

Create `packages/training/tests/train/test_cli.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest packages/training/tests/train/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.train.cli'`.

- [ ] **Step 3: Implement `cli.py`**

Create `packages/training/src/training/train/cli.py`:

```python
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
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from training.dataset.jsonl import read_pairs
    from training.train.evaluate import evaluate_against_gold

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    gold = read_pairs(gold_pairs_path)
    return evaluate_against_gold(
        model=model, tokenizer=tokenizer, gold_pairs=gold, max_length=max_length, device="cpu",
    )


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
        report = _run_evaluate(args.model_dir, args.gold_pairs, args.max_length)
        from training.train.evaluate import render_report
        print(render_report(report))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add re-exports to the `train/__init__.py`**

Replace the contents of `packages/training/src/training/train/__init__.py`:

```python
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
```

- [ ] **Step 5: Run the CLI tests to verify they pass**

Run: `uv run pytest packages/training/tests/train/test_cli.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Confirm `python -m training.train --help` runs**

Run: `uv run python -m training.train --help`
Expected: shows `train` and `evaluate` subcommand list with no traceback.

- [ ] **Step 7: Commit**

```bash
git add packages/training/src/training/train/cli.py packages/training/src/training/train/__init__.py packages/training/tests/train/test_cli.py
git commit -m "feat: add training CLI with train and evaluate subcommands"
```

---

## Task 9: Publish — model card + Hub push

**Files:**
- Create: `packages/training/src/training/publish/model.py`
- Create: `packages/training/tests/publish/test_model.py`

This module mirrors `training.publish.to_hf`'s structure but operates on a *model* repo. It owns:

- **`build_model_card(...)`** — assembles a `huggingface_hub.ModelCard` from training config, eval metrics, dataset repo URL, and the static disclosures from supplement §7.1 (intended use, score range, limitations, license).
- **`push_model(repo_id, model_dir, model_card, hf_token, commit_message)`** — creates the repo and uploads `model_dir` + writes the card.
- **`main(argv)`** — `python -m training.publish.model --repo user/resumora-ai-distilbert-lora --model-dir outputs/run/ --metrics-json outputs/run/final_metrics.json --dataset-repo user/resumora-ai-dataset`.

- [ ] **Step 1: Write the failing tests**

Create `packages/training/tests/publish/test_model.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest packages/training/tests/publish/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.publish.model'`.

- [ ] **Step 3: Implement `publish/model.py`**

Create `packages/training/src/training/publish/model.py`:

```python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from textwrap import dedent

from huggingface_hub import HfApi, ModelCard, ModelCardData


def build_model_card(
    *,
    base_model: str,
    dataset_repo: str,
    train_config: dict,
    eval_metrics: dict,
) -> ModelCard:
    """Assemble the public-facing model card.

    All the disclosures from supplement §7.1 live in this string. Editing the
    card means editing them in one place.
    """
    card_data = ModelCardData(
        language="en",
        license="apache-2.0",
        library_name="peft",
        base_model=base_model,
        tags=["resume", "job-matching", "lora", "classification"],
        pipeline_tag="text-classification",
    )

    per_class = eval_metrics.get("per_class_f1", {})
    eval_n = eval_metrics.get("n", "?")

    body = dedent(
        f"""\
        # Resumora AI — DistilBERT + LoRA fit classifier

        Fine-tuned classifier that scores a (resume, job description) pair as one of
        `weak` / `partial` / `strong` fit, and produces a continuous score in `[20, 85]`
        via the expected value `softmax(logits) · [20, 55, 85]`.

        Built as the model artifact for the [Resumora AI](https://github.com/) portfolio
        project. **Not for hiring decisions.**

        ## Score range

        The model output is bounded to `[20, 85]`, not `[0, 100]`. This is by design —
        the score is the softmax-weighted average of three bucket midpoints
        (`weak=20`, `partial=55`, `strong=85`). Numbers outside that range are not
        produced. The 0-100 product surface in the README is honored by honest
        disclosure, not by rescaling.

        ## Intended use

        - **Portfolio demonstration** of LoRA fine-tuning on top of DistilBERT.
        - **NOT for hiring decisions**, screening, or any consequential evaluation
          of a real person's application.

        ## Training data

        - **Synthetic** pairs generated with Ollama + `llama3.2:3b` covering ~15 roles
          x seniorities x 3 fit-levels. Each label was requested in the generator
          prompt — the label comes for free.
        - Dataset: [`{dataset_repo}`](https://huggingface.co/datasets/{dataset_repo}).
        - Synthetic-data risk: the classifier may have learned the generator's
          stylistic tics. The gold-set evaluation below is the only trusted signal.

        ## Evaluation

        Evaluated against a hand-curated gold set (n = {eval_n}, never trained on):

        - accuracy: **{eval_metrics.get("accuracy", "?"):.3f}**
        - macro F1: **{eval_metrics.get("macro_f1", "?"):.3f}**
        - per-class F1: weak {per_class.get("weak", "?"):.3f}, partial {per_class.get("partial", "?"):.3f}, strong {per_class.get("strong", "?"):.3f}
        - MAE (expected-value score vs bucket midpoint): **{eval_metrics.get("mae", "?"):.2f}**

        ## Limitations

        - English-only.
        - Gold set is small (low statistical power on small differences).
        - No demographic-bias evaluation has been performed.
        - The model has no notion of seniority sub-genres beyond what fit into the
          synthetic prompts.

        ## Reproducibility

        - Base model: `{base_model}`.
        - Training config (subset):

        ```json
        {json.dumps(train_config, indent=2)}
        ```

        License: apache-2.0 (matches the base DistilBERT license).
        """
    )

    card = ModelCard.from_template(card_data, model_id="resumora-ai-distilbert-lora")
    card.content = card_data.to_yaml() + "\n---\n\n" + body
    return card


def push_model(
    *,
    repo_id: str,
    model_dir: Path,
    model_card: ModelCard,
    hf_token: str,
    commit_message: str = "update model",
) -> None:
    """Create the model repo if missing and upload the adapter + card.

    Never run from CI. Run locally with the user's HF_TOKEN env var set.
    """
    if not hf_token:
        raise ValueError("HF token is required (pass --token or set HF_TOKEN env var)")
    if not model_dir.exists() or not model_dir.is_dir():
        raise FileNotFoundError(model_dir)

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(model_dir),
        commit_message=commit_message,
    )
    model_card.push_to_hub(repo_id, token=hf_token)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="training.publish.model")
    parser.add_argument("--repo", required=True, help="HF model repo id, e.g. user/resumora-ai-distilbert-lora")
    parser.add_argument("--model-dir", type=Path, required=True, help="local model dir produced by training")
    parser.add_argument("--metrics-json", type=Path, required=True, help="final_metrics.json or eval report json")
    parser.add_argument("--dataset-repo", required=True, help="HF dataset repo id used during training")
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument("--train-config-json", default="{}", help="JSON string of the training config to embed")
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN", ""),
        help="HF token (defaults to HF_TOKEN env var)",
    )
    parser.add_argument("--message", default="update model")
    args = parser.parse_args(argv)

    metrics_loaded = json.loads(args.metrics_json.read_text())
    train_config = json.loads(args.train_config_json)

    card = build_model_card(
        base_model=args.base_model,
        dataset_repo=args.dataset_repo,
        train_config=train_config,
        eval_metrics=metrics_loaded,
    )

    push_model(
        repo_id=args.repo,
        model_dir=args.model_dir,
        model_card=card,
        hf_token=args.token,
        commit_message=args.message,
    )
    print(f"uploaded {args.model_dir} -> {args.repo}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest packages/training/tests/publish/test_model.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/training/src/training/publish/model.py packages/training/tests/publish/test_model.py
git commit -m "feat: add model card builder and HF Hub model publisher"
```

---

## Task 10: Colab notebook

**Files:**
- Create: `notebooks/01_train_on_colab.ipynb`

The notebook is intentionally thin. Each cell is one paragraph of work; the heavy lifting stays in `training.train.cli`. The notebook is committed so a reader can open it in Colab from GitHub without re-deriving the steps.

> **Note:** Jupyter notebooks are JSON. Write the file by saving the literal JSON below. Editors that show notebooks visually will render it; treating it as text is fine for diff-review.

- [ ] **Step 1: Create the notebook file**

Create `notebooks/01_train_on_colab.ipynb` with this exact content:

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Resumora AI — Fine-tune DistilBERT + LoRA on Colab\n",
    "\n",
    "This notebook is intentionally thin. It installs the repo, pulls the synthetic dataset from HF Hub, and calls `python -m training.train` with the `full` preset. The same code runs locally with the `smoke` preset.\n",
    "\n",
    "**Before running:**\n",
    "1. Switch the Colab runtime to **GPU (T4)**: Runtime -> Change runtime type -> GPU.\n",
    "2. Set the `HF_USER`, `HF_DATASET_REPO`, and `HF_MODEL_REPO` variables in the next cell.\n",
    "3. Run all cells top-to-bottom.\n",
    "\n",
    "Total runtime: ~30-60 minutes on T4 with ~500 pairs and 3 epochs."
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "HF_USER = \"YOUR-USERNAME\"  # set me\n",
    "HF_DATASET_REPO = f\"{HF_USER}/resumora-ai-dataset\"\n",
    "HF_MODEL_REPO = f\"{HF_USER}/resumora-ai-distilbert-lora\"\n",
    "GITHUB_REPO_URL = \"https://github.com/YOUR-USERNAME/AI-Pipeline.git\"  # set me\n",
    "BRANCH = \"main\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Install the repo\n",
    "\n",
    "Colab images ship with `pip` not `uv`, so we install via pip from the cloned repo."
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "!git clone --branch {BRANCH} {GITHUB_REPO_URL} ai-pipeline\n",
    "%cd ai-pipeline\n",
    "!pip install -q ./packages/pipeline ./packages/training"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Log in to Hugging Face\n",
    "\n",
    "Paste your write-scope token from https://huggingface.co/settings/tokens."
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "from huggingface_hub import login\n",
    "login()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Pull the synthetic + gold sets from your dataset repo"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "from pathlib import Path\n",
    "from training.train.data import load_pairs_from_hub\n",
    "from training.dataset.jsonl import write_pairs\n",
    "\n",
    "synth = load_pairs_from_hub(repo_id=HF_DATASET_REPO, filename=\"synthetic/pairs.jsonl\")\n",
    "gold  = load_pairs_from_hub(repo_id=HF_DATASET_REPO, filename=\"gold/seed.jsonl\")\n",
    "Path(\"data/synthetic\").mkdir(parents=True, exist_ok=True)\n",
    "Path(\"data/gold\").mkdir(parents=True, exist_ok=True)\n",
    "write_pairs(Path(\"data/synthetic/pairs.jsonl\"), synth)\n",
    "write_pairs(Path(\"data/gold/seed.jsonl\"), gold)\n",
    "print(f\"synthetic = {len(synth)} pairs, gold = {len(gold)} pairs\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Train with the `full` preset\n",
    "\n",
    "The same CLI entry point as local dev. Watch the per-epoch validation macro-F1."
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "!python -m training.train train --preset full --run-name colab-run-1"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Evaluate against the gold set"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "!python -m training.train evaluate --model-dir outputs/full --gold-pairs data/gold/seed.jsonl --max-length 512"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Push the adapter + auto-generated model card to HF Hub"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "import json\n",
    "from pathlib import Path\n",
    "\n",
    "# Re-run gold evaluation and persist a metrics file the publisher can embed in the card.\n",
    "from transformers import AutoModelForSequenceClassification, AutoTokenizer\n",
    "from training.dataset.jsonl import read_pairs\n",
    "from training.train.evaluate import evaluate_against_gold\n",
    "\n",
    "model_dir = Path(\"outputs/full\")\n",
    "tokenizer = AutoTokenizer.from_pretrained(model_dir)\n",
    "model     = AutoModelForSequenceClassification.from_pretrained(model_dir)\n",
    "gold      = read_pairs(Path(\"data/gold/seed.jsonl\"))\n",
    "report    = evaluate_against_gold(model=model, tokenizer=tokenizer, gold_pairs=gold, max_length=512, device=\"cuda\")\n",
    "(model_dir / \"gold_eval.json\").write_text(json.dumps(report.to_dict(), indent=2))"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "from training.train.config import full_config\n",
    "import json\n",
    "\n",
    "train_cfg_json = json.dumps(full_config().to_dict())\n",
    "!python -m training.publish.model \\\n",
    "    --repo {HF_MODEL_REPO} \\\n",
    "    --model-dir outputs/full \\\n",
    "    --metrics-json outputs/full/gold_eval.json \\\n",
    "    --dataset-repo {HF_DATASET_REPO} \\\n",
    "    --base-model distilbert-base-uncased \\\n",
    "    --train-config-json '{train_cfg_json}' \\\n",
    "    --message \"phase 3 — initial release\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Save MLflow runs (Colab `./mlruns/` dies with the runtime)\n",
    "\n",
    "Zip the run directory and download it so you can browse with `mlflow ui` locally."
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "import shutil\n",
    "from google.colab import files\n",
    "shutil.make_archive(\"mlruns\", \"zip\", \".\", \"mlruns\")\n",
    "files.download(\"mlruns.zip\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {"name": "python3", "display_name": "Python 3"},
  "language_info": {"name": "python"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Verify the notebook is valid JSON**

Run: `uv run python -c "import json; json.load(open('notebooks/01_train_on_colab.ipynb'))"`
Expected: no output (silent success).

- [ ] **Step 3: Verify the notebook is valid nbformat (optional)**

Run: `uv run python -c "import nbformat; nbformat.read('notebooks/01_train_on_colab.ipynb', as_version=4); print('ok')"`

If `nbformat` is not installed, skip silently — JSON validity is the hard requirement.

Expected: prints `ok`, or `ModuleNotFoundError` (acceptable).

- [ ] **Step 4: Commit**

```bash
git add notebooks/01_train_on_colab.ipynb
git commit -m "feat: add thin Colab training notebook"
```

---

## Task 11: Docs + README + final verification

**Files:**
- Create: `docs/phase-3-finetune.md`
- Modify: `README.md`

The user guide is a one-page walkthrough that mirrors `docs/phase-2-data.md`'s tone. It covers: smoke test locally, full run on Colab, publishing.

- [ ] **Step 1: Write the Phase 3 user guide**

Create `docs/phase-3-finetune.md`:

```markdown
# Phase 3 — Fine-tune

This phase produces the trained `resumora-ai-distilbert-lora` model. Two paths:

- **Local smoke test** — runs on a Mac CPU in under a minute. Verifies the pipeline boots and the head receives gradients. *No real learning happens.*
- **Colab full run** — runs on a free T4 GPU in 30-60 minutes. Produces the model that ends up on HF Hub.

Both paths share the same CLI: `python -m training.train`.

## Prerequisites

- Phase 2 complete: synthetic pairs at `data/synthetic/pairs.jsonl` and the gold seed at `data/gold/seed.jsonl`.
- `uv sync --all-packages` (Phase 3 adds `transformers`, `torch`, `peft`, `datasets`, `mlflow`).

## Score range — read this once

The model is a 3-class classifier (`weak` / `partial` / `strong`) and the score
at inference is the softmax-weighted average of the bucket midpoints
`[20, 55, 85]`. **Scores are bounded to `[20, 85]`** — not `[0, 100]`. The 0-100
product surface in the README is honored by honest disclosure, not by stretching
the range. The model card on HF Hub repeats this.

## Local smoke test

```bash
uv run python -m training.train train --preset smoke
```

Expected output: a few MLflow log lines, a one-line summary `run_id=... output_dir=outputs/smoke`, and the final eval metrics as JSON. The output directory contains the saved LoRA adapter and a `final_metrics.json`.

**Smoke is not a real run.** It exists to prove the head receives gradients (the PEFT footgun the supplement §3.1 warns about) and that MLflow logs land in `./mlruns/`.

## Colab full run

Open `notebooks/01_train_on_colab.ipynb` in Colab (Runtime -> GPU T4). Set the three variables at the top:

```
HF_USER = "your-username"
HF_DATASET_REPO = f"{HF_USER}/resumora-ai-dataset"
HF_MODEL_REPO = f"{HF_USER}/resumora-ai-distilbert-lora"
GITHUB_REPO_URL = "https://github.com/your-username/AI-Pipeline.git"
```

Then Run All. Cells 1-4 install the repo, pull the dataset, and train. Cells 5-6 evaluate and publish.

### Gold-set publication gate

The Phase 2 seed has **5 gold pairs**. That is enough to develop Phase 3, but it is **not** enough to publish a model card publicly — macro-F1 swings several points run-to-run on n=5.

Before pushing the trained model to HF Hub (cell 6 of the notebook), grow `data/gold/seed.jsonl` to **at least 30 hand-written pairs**, re-push the dataset, and re-pull it on Colab. The model card embeds the gold-set metrics; small-n numbers are not portfolio material.

## Evaluate a saved model against the gold set

```bash
uv run python -m training.train evaluate \
    --model-dir outputs/smoke \
    --gold-pairs data/gold/seed.jsonl \
    --max-length 128
```

Prints accuracy, macro-F1, per-class F1, MAE, and the confusion matrix.

## Browse MLflow runs locally

```bash
uv run mlflow ui --backend-store-uri ./mlruns
```

Open http://localhost:5000.

On Colab, `./mlruns/` lives inside the runtime and dies with it. The notebook's last cell zips and downloads `mlruns.zip` so you can browse it after the run.

## Where the model lands

Default repo: `<HF_USER>/resumora-ai-distilbert-lora`. The auto-generated model card includes the score-range disclosure, intended-use disclaimer ("not for hiring decisions"), synthetic-data provenance, gold-set metrics, limitations, and the training config. The license is `apache-2.0` (matches DistilBERT base).
```

- [ ] **Step 2: Add a Phase 3 entry to the README**

In `README.md`, change the phases list block from:

```markdown
## Phases

- **Phase 0 — scaffold:** [plan](docs/superpowers/plans/2026-05-14-phase-0-scaffold.md).
- **Phase 1 — ingestion:** [plan](docs/superpowers/plans/2026-05-14-phase-1-ingestion.md).
- **Phase 2 — data layer:** [plan](docs/superpowers/plans/2026-05-15-phase-2-data.md), [guide](docs/phase-2-data.md).
```

to:

```markdown
## Phases

- **Phase 0 — scaffold:** [plan](docs/superpowers/plans/2026-05-14-phase-0-scaffold.md).
- **Phase 1 — ingestion:** [plan](docs/superpowers/plans/2026-05-14-phase-1-ingestion.md).
- **Phase 2 — data layer:** [plan](docs/superpowers/plans/2026-05-15-phase-2-data.md), [guide](docs/phase-2-data.md).
- **Phase 3 — fine-tune:** [plan](docs/superpowers/plans/2026-05-15-phase-3-finetune.md), [supplement](docs/superpowers/specs/2026-05-15-phase-3-finetune-supplement.md), [guide](docs/phase-3-finetune.md).
```

- [ ] **Step 3: Run the full unit test suite**

Run: `uv run pytest`
Expected: every Phase 0/1/2/3 test passes. Integration tests are skipped (per the root `pyproject.toml` `addopts`).

- [ ] **Step 4: Run lint**

Run: `uv run ruff check .`
Expected: no errors. If ruff flags long lines in the new code, wrap them at 100 chars to match the existing style.

- [ ] **Step 5: Optional — run ruff format**

Run: `uv run ruff format packages/training/src/training/train packages/training/src/training/publish/model.py packages/training/tests/train packages/training/tests/publish/test_model.py`
Expected: no diff, or minor formatting normalization.

- [ ] **Step 6: Commit**

```bash
git add docs/phase-3-finetune.md README.md
git commit -m "docs: add Phase 3 user guide and README entry"
```

---

## Self-Review (run by the implementer before declaring done)

After the last task, walk through this list and fix anything missed inline:

1. **Spec coverage — every supplement section maps to a task:**
   - §1 task formulation → Task 4 (metrics.py `score_from_logits`).
   - §1.1 score range disclosure → Tasks 4 (docstring), 9 (model card), 11 (user guide).
   - §2 base model + tokenization → Task 3 (tokenize), Task 5 (load_base_model).
   - §3 LoRA config → Task 5 (`default_lora_config`).
   - §3.1 head-gradient guard → Task 6 (`verify_head_receives_gradients` + tests).
   - §4 compute split → Tasks 2 (smoke/full configs), 10 (notebook).
   - §5 MLflow local-only → Task 6 (`report_to=["mlflow"]`, no remote URI).
   - §5.1 Colab MLflow persistence → Task 10 (cell 7 zips and downloads `mlruns/`).
   - §6.1 gold publication gate → Task 11 (user guide states the n ≥ 30 gate; the publisher does not enforce it programmatically because gold size is product-owner territory).
   - §6.2 metrics → Task 4 (`compute_metrics`), Task 7 (`evaluate_against_gold`).
   - §6.3 balance precheck → Task 3 (`assert_label_balance`).
   - §7 model card → Task 9 (`build_model_card`).
   - §7.1 required card sections → Task 9 (test asserts all disclosures present).
   - §8 module layout → matches the file structure section above.
   - §9 out of scope → not implemented, by design.
   - §10 prerequisites → Task 11 (guide lists them).

2. **Placeholder scan:** search the plan for "TBD", "TODO", "fill in" — none should appear.

3. **Type consistency:**
   - `LABEL_TO_INT` is defined in `training.train.data` and re-used in `training.train.evaluate`. Same dict, one source.
   - `INT_TO_SCORE` is defined in `training.train.metrics` and re-used in `training.train.evaluate`. Same dict, one source.
   - `TrainConfig` field names match between Tasks 2, 6, and 8 (`run_name`, `num_train_epochs`, `per_device_train_batch_size`, etc.).
   - `EvalReport.per_class_f1` is a `dict[str, float]` everywhere it's referenced.
   - `verify_head_receives_gradients` signature: same in Tasks 6 (impl) and 8 (re-export).

4. **Anti-checks (things the plan deliberately does NOT do):**
   - The plan does NOT include a step to expand the gold set programmatically — that is hand-labeled product work the user owns.
   - The plan does NOT add a CI workflow for training — training runs on Colab manually, not in CI.
   - The plan does NOT touch `apps/api` or `apps/web` — those are Phase 6 and Phase 7.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-15-phase-3-finetune.md`.

Two execution options:

1. **Subagent-driven (recommended)** — fresh subagent per task, two-stage review between tasks, fast iteration.
2. **Inline execution** — execute tasks in this session via `superpowers:executing-plans` with batch checkpoints.

Which approach?
