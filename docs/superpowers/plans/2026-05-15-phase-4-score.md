# Phase 4 — Score & Similarity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `packages/pipeline/src/pipeline/scoring/` and `packages/pipeline/src/pipeline/similarity/` — two pure libraries that, together, turn a `(ResumeProfile + resume_text, JobProfile + jd_text)` input into a fit score (Phase 3 model) plus a matched/missing skills report (sentence-transformers).

**Architecture:** Both modules are pure Python sub-packages with no FastAPI, no Ollama, no orchestration knowledge. `pipeline.scoring` loads the Phase 3 DistilBERT+LoRA adapter eagerly via `Scorer.from_pretrained(repo_id_or_path)`, then `scorer.score(resume_text, jd_text)` returns a `ScoreResult`. `pipeline.similarity` exposes `SkillMatcher.from_pretrained(model)`; `matcher.match(resume_profile, job_profile)` returns a `SkillMatchReport` of required- and nice-to-have-skill matches based on cosine similarity over MiniLM embeddings with a default threshold of `0.55`. Neither module imports from `packages/training`; constants are re-declared in `pipeline.scoring._math` and pinned by tests.

**Tech Stack:** Python 3.12, `transformers`, `peft`, `torch` (CPU default), `sentence-transformers`, `numpy`, `pydantic`. Tests use `pytest` with the existing `not integration` default marker pattern. Hermetic unit tests build a tiny DistilBERT from `DistilBertConfig` (Phase 3 pattern) and inject a fake `EmbeddingBackend` (Protocol).

**This plan is Phase 4 only.** It follows the master design doc `docs/superpowers/specs/2026-05-14-ai-pipeline-design.md` (§4 components 3+4, §7 phase 4) and its supplement `docs/superpowers/specs/2026-05-15-phase-4-score-supplement.md`. Decisions locked in during brainstorming on 2026-05-15:

- **Two libraries, one phase** — `pipeline.scoring` (load Phase 3 model, classify) and `pipeline.similarity` (skill embeddings + match report).
- **Score range `[20, 85]` is inherited from Phase 3** — `Scorer.score` docstring and `ScoreResult.score` description state this; no rescaling.
- **`pipeline` does not depend on `training`** — constants and softmax/score math are re-declared in `pipeline/scoring/_math.py` and pinned by unit tests against the supplement values.
- **Default similarity threshold `0.55`** — middle of the band for MiniLM cosine on short skill phrases; constructor accepts an override; documented as tunable.
- **`match_rate = 1.0` when `required_skills` is empty** — documented vacuous truth (vs `NaN`, which would force every caller to special-case).
- **Embedding skill phrases are lower-cased and stripped, not de-punctuated** — `node.js` stays `node.js`.
- **Integration tests gated** — anything that touches a real HF Hub model is `@pytest.mark.integration`, consistent with the Ollama + HF Hub pattern from Phases 2 and 3.

> **Prerequisite:** A Phase 3 model exists either on HF Hub (`<HF_USERNAME>/resumora-ai-distilbert-lora`) or as a locally saved adapter directory. Phase 4 development can proceed against a local smoke-trained adapter. The only step that *requires* the Hub model is the gated integration test (Task 5 step 9).

> **Note on PyTorch inference-mode method:** As in the Phase 3 plan, this plan calls `model.train(False)` everywhere — never the shorter-named method. Identical behavior; avoids a token that a security linter could confuse with Python's builtin code-execution function.

---

## File Structure

Files created or modified in this phase:

- `packages/pipeline/pyproject.toml` — **modify**: add `transformers`, `torch`, `peft`, `sentence-transformers`.
- `packages/pipeline/src/pipeline/scoring/__init__.py` — **create**: public surface (`Scorer`, `ScoreResult`).
- `packages/pipeline/src/pipeline/scoring/_math.py` — **create**: bucket constants, softmax, score/confidence helpers.
- `packages/pipeline/src/pipeline/scoring/models.py` — **create**: `ScoreResult` pydantic model + `PREDICTED_LABELS`.
- `packages/pipeline/src/pipeline/scoring/loader.py` — **create**: `load_scorer_artifacts()` — base model + tokenizer + PEFT adapter.
- `packages/pipeline/src/pipeline/scoring/scorer.py` — **create**: `Scorer` class.
- `packages/pipeline/src/pipeline/similarity/__init__.py` — **create**: public surface (`SkillMatcher`, `SkillMatchReport`, `SkillMatch`).
- `packages/pipeline/src/pipeline/similarity/models.py` — **create**: `SkillMatch`, `SkillMatchReport` pydantic models.
- `packages/pipeline/src/pipeline/similarity/_embeddings.py` — **create**: `EmbeddingBackend` Protocol + `SentenceTransformerBackend` default impl.
- `packages/pipeline/src/pipeline/similarity/matcher.py` — **create**: `SkillMatcher` class.
- `packages/pipeline/tests/scoring/__init__.py` — **create**: empty marker.
- `packages/pipeline/tests/scoring/test_math.py` — **create**: pin bucket constants, verify `[20, 85]` range, softmax shape.
- `packages/pipeline/tests/scoring/test_models.py` — **create**: tests for `ScoreResult` validation.
- `packages/pipeline/tests/scoring/test_loader.py` — **create**: tests against a tiny config-built DistilBERT; integration test for HF Hub.
- `packages/pipeline/tests/scoring/test_scorer.py` — **create**: tests for `Scorer.score` end-to-end on a tiny model.
- `packages/pipeline/tests/similarity/__init__.py` — **create**: empty marker.
- `packages/pipeline/tests/similarity/test_models.py` — **create**: tests for `SkillMatch` / `SkillMatchReport`.
- `packages/pipeline/tests/similarity/test_embeddings.py` — **create**: tests for `EmbeddingBackend` protocol + fake backend; integration test for real backend.
- `packages/pipeline/tests/similarity/test_matcher.py` — **create**: tests for `SkillMatcher.match` (threshold, empty cases, match_rate).
- `docs/phase-4-score.md` — **create**: user-facing Phase 4 guide.
- `README.md` — **modify**: add Phase 4 entry to the phases list.

---

## Task 1: Dependencies and scoring sub-package skeleton

**Files:**
- Modify: `packages/pipeline/pyproject.toml`
- Create: `packages/pipeline/src/pipeline/scoring/__init__.py`
- Create: `packages/pipeline/tests/scoring/__init__.py`

- [ ] **Step 1: Add Phase 4 runtime dependencies**

In `packages/pipeline/pyproject.toml`, change the `dependencies` list from:

```toml
dependencies = [
    "pypdf>=5.0",
    "python-docx>=1.1",
    "pydantic>=2.0",
    "httpx>=0.27",
]
```

to:

```toml
dependencies = [
    "pypdf>=5.0",
    "python-docx>=1.1",
    "pydantic>=2.0",
    "httpx>=0.27",
    "transformers>=4.44",
    "torch>=2.4",
    "peft>=0.13",
    "sentence-transformers>=3.0",
    "numpy>=1.26",
]
```

- [ ] **Step 2: Create the empty scoring sub-package marker**

Create `packages/pipeline/src/pipeline/scoring/__init__.py` with this exact content (re-exports will be added in later tasks):

```python
"""Phase 4 scoring library: load the fine-tuned DistilBERT+LoRA model and score (resume, JD) pairs."""
```

- [ ] **Step 3: Create the empty scoring tests marker**

Create `packages/pipeline/tests/scoring/__init__.py` as a zero-byte file (the marker pytest needs to discover the sub-package).

- [ ] **Step 4: Sync the workspace**

Run: `uv sync --all-packages`
Expected: completes without error; new dependencies are installed; `uv.lock` is updated. The install may take 1–2 minutes the first time because of `torch` and `sentence-transformers`.

- [ ] **Step 5: Verify the new dependencies import**

Run:
```bash
uv run python -c "import transformers, torch, peft, sentence_transformers, numpy; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 6: Commit**

```bash
git add packages/pipeline/pyproject.toml packages/pipeline/src/pipeline/scoring/__init__.py packages/pipeline/tests/scoring/__init__.py uv.lock
git commit -m "feat: add Phase 4 pipeline runtime dependencies and scoring skeleton"
```

---

## Task 2: Scoring math helpers

**Files:**
- Create: `packages/pipeline/src/pipeline/scoring/_math.py`
- Create: `packages/pipeline/tests/scoring/test_math.py`

`_math.py` re-declares the Phase 3 inference constants and provides softmax / expected-value / confidence helpers. It is the *single source of truth* inside `pipeline` for the `[20, 85]` range — Phase 6 and Phase 7 read from here.

- [ ] **Step 1: Write the failing tests**

Create `packages/pipeline/tests/scoring/test_math.py`:

```python
import numpy as np

from pipeline.scoring._math import (
    BUCKET_SCORES,
    INT_TO_LABEL,
    confidence_from_logits,
    score_from_logits,
    softmax,
)


def test_bucket_scores_match_phase3_supplement():
    # Pinned to the Phase 3 supplement §1: the actual range is [20, 85].
    assert BUCKET_SCORES == [20.0, 55.0, 85.0]


def test_int_to_label_mapping_is_canonical():
    assert INT_TO_LABEL == {0: "weak", 1: "partial", 2: "strong"}


def test_softmax_rows_sum_to_one():
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    probs = softmax(logits)
    np.testing.assert_allclose(probs.sum(axis=-1), [1.0, 1.0], atol=1e-7)


def test_score_from_logits_is_bounded_to_20_85():
    # Confident weak → near 20; confident strong → near 85; nothing escapes the range.
    logits = np.array([
        [10.0, 0.0, 0.0],   # weak
        [0.0, 0.0, 10.0],   # strong
        [0.0, 10.0, 0.0],   # partial
    ])
    scores = score_from_logits(logits)
    assert scores[0] < 25
    assert scores[1] > 80
    assert 50 < scores[2] < 60
    for s in scores:
        assert 20.0 <= s <= 85.0


def test_score_from_logits_uniform_is_average_bucket():
    logits = np.zeros((1, 3))
    scores = score_from_logits(logits)
    assert abs(scores[0] - (20 + 55 + 85) / 3) < 1e-9


def test_confidence_from_logits_is_max_prob():
    logits = np.array([[10.0, 0.0, 0.0]])
    conf = confidence_from_logits(logits)
    assert conf[0] > 0.99


def test_score_from_logits_accepts_lists():
    # Convenience: callers may pass plain Python lists from non-numpy code.
    scores = score_from_logits([[0.0, 0.0, 0.0]])
    assert abs(scores[0] - (20 + 55 + 85) / 3) < 1e-9
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest packages/pipeline/tests/scoring/test_math.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.scoring._math'`.

- [ ] **Step 3: Implement `_math.py`**

Create `packages/pipeline/src/pipeline/scoring/_math.py`:

```python
from __future__ import annotations

import numpy as np

# Pinned to the Phase 3 supplement §1. Re-declared (not imported from training)
# so pipeline stays training-independent at runtime; tests pin the values.
BUCKET_SCORES: list[float] = [20.0, 55.0, 85.0]
INT_TO_LABEL: dict[int, str] = {0: "weak", 1: "partial", 2: "strong"}

_SCORE_VECTOR = np.array(BUCKET_SCORES)


def softmax(logits: np.ndarray | list) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64)
    shifted = arr - arr.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def score_from_logits(logits: np.ndarray | list) -> np.ndarray:
    """Expected-value score: softmax(logits) · [20, 55, 85].

    Output is bounded to [20.0, 85.0] — this is the deliberate Phase 3 range
    (see Phase 3 supplement §1.1). The product-surface "0–100" is honored by
    disclosure, not by stretching the range.
    """
    probs = softmax(logits)
    return probs @ _SCORE_VECTOR


def confidence_from_logits(logits: np.ndarray | list) -> np.ndarray:
    """Max softmax probability — a simple per-prediction confidence in [0, 1]."""
    probs = softmax(logits)
    return probs.max(axis=-1)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest packages/pipeline/tests/scoring/test_math.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/pipeline/src/pipeline/scoring/_math.py packages/pipeline/tests/scoring/test_math.py
git commit -m "feat: add scoring math helpers with pinned [20, 85] range"
```

---

## Task 3: `ScoreResult` pydantic model

**Files:**
- Create: `packages/pipeline/src/pipeline/scoring/models.py`
- Create: `packages/pipeline/tests/scoring/test_models.py`

`ScoreResult` is a validated Pydantic model that the `Scorer` returns. It documents the `[20, 85]` range in its field description and constrains the per-class probabilities to the canonical three keys.

- [ ] **Step 1: Write the failing tests**

Create `packages/pipeline/tests/scoring/test_models.py`:

```python
import pytest
from pydantic import ValidationError

from pipeline.scoring.models import PREDICTED_LABELS, ScoreResult


def test_predicted_labels_constant():
    assert PREDICTED_LABELS == ("weak", "partial", "strong")


def test_score_result_valid():
    r = ScoreResult(
        score=72.3,
        confidence=0.81,
        class_probabilities={"weak": 0.05, "partial": 0.14, "strong": 0.81},
        predicted_label="strong",
    )
    assert r.score == 72.3
    assert r.predicted_label == "strong"


def test_score_result_score_below_20_rejected():
    with pytest.raises(ValidationError):
        ScoreResult(
            score=15.0,
            confidence=0.5,
            class_probabilities={"weak": 0.6, "partial": 0.3, "strong": 0.1},
            predicted_label="weak",
        )


def test_score_result_score_above_85_rejected():
    with pytest.raises(ValidationError):
        ScoreResult(
            score=90.0,
            confidence=0.5,
            class_probabilities={"weak": 0.1, "partial": 0.3, "strong": 0.6},
            predicted_label="strong",
        )


def test_score_result_invalid_label_rejected():
    with pytest.raises(ValidationError):
        ScoreResult(
            score=50.0,
            confidence=0.5,
            class_probabilities={"weak": 0.4, "partial": 0.4, "strong": 0.2},
            predicted_label="excellent",  # not in PREDICTED_LABELS
        )


def test_score_result_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        ScoreResult(
            score=50.0,
            confidence=1.4,
            class_probabilities={"weak": 0.4, "partial": 0.4, "strong": 0.2},
            predicted_label="weak",
        )


def test_score_result_missing_class_key_rejected():
    with pytest.raises(ValidationError):
        ScoreResult(
            score=50.0,
            confidence=0.5,
            class_probabilities={"weak": 0.5, "partial": 0.5},  # missing strong
            predicted_label="weak",
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest packages/pipeline/tests/scoring/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.scoring.models'`.

- [ ] **Step 3: Implement `models.py`**

Create `packages/pipeline/src/pipeline/scoring/models.py`:

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

PredictedLabel = Literal["weak", "partial", "strong"]
PREDICTED_LABELS: tuple[PredictedLabel, ...] = ("weak", "partial", "strong")


class ScoreResult(BaseModel):
    """Output of `Scorer.score(resume_text, jd_text)`.

    `score` is bounded to [20.0, 85.0] because the underlying classifier emits
    a softmax-weighted average of bucket midpoints [20, 55, 85]. See the Phase 3
    supplement §1.1 for the rationale.
    """

    score: float = Field(
        ge=20.0,
        le=85.0,
        description="Expected-value fit score in [20.0, 85.0].",
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Max softmax probability.")
    class_probabilities: dict[str, float] = Field(
        description="Probabilities for {weak, partial, strong}; values in [0, 1] sum to ~1.",
    )
    predicted_label: PredictedLabel

    @field_validator("class_probabilities")
    @classmethod
    def _exact_three_keys(cls, value: dict[str, float]) -> dict[str, float]:
        expected = set(PREDICTED_LABELS)
        if set(value.keys()) != expected:
            raise ValueError(
                f"class_probabilities must have keys {sorted(expected)}; got {sorted(value.keys())}"
            )
        for k, v in value.items():
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"class_probabilities[{k!r}] = {v} not in [0, 1]")
        return value
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest packages/pipeline/tests/scoring/test_models.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/pipeline/src/pipeline/scoring/models.py packages/pipeline/tests/scoring/test_models.py
git commit -m "feat: add ScoreResult pydantic model with [20, 85] bound"
```

---

## Task 4: Scorer artifact loader

**Files:**
- Create: `packages/pipeline/src/pipeline/scoring/loader.py`
- Create: `packages/pipeline/tests/scoring/test_loader.py`

`loader.py` returns a `(model, tokenizer)` pair given a `repo_id_or_path`. It:

1. Tries to load the tokenizer from `repo_id_or_path`; falls back to the base model's tokenizer if that fails (PEFT adapters often skip publishing the tokenizer).
2. Loads the base model (`AutoModelForSequenceClassification` with `num_labels=3`) and applies the PEFT adapter on top via `PeftModel.from_pretrained`.
3. Moves the model to `device` and calls `model.train(False)` for inference mode.

Unit tests use a config-built tiny DistilBERT saved to a temp directory and **do not** apply a PEFT adapter (PEFT load is exercised by an integration test). Tests inject the tiny base model path directly to avoid network.

- [ ] **Step 1: Write the failing tests**

Create `packages/pipeline/tests/scoring/test_loader.py`:

```python
from pathlib import Path

import pytest
import torch
from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizerFast

from pipeline.scoring.loader import load_scorer_artifacts


def _make_tiny_base(tmp_path: Path) -> Path:
    """Build a tiny DistilBERT-for-seq-classification and save it to disk.

    The tokenizer is hand-built from a 100-token vocab so no download is needed.
    """
    cfg = DistilBertConfig(
        vocab_size=200,
        max_position_embeddings=64,
        num_hidden_layers=1,
        n_layers=1,
        n_heads=2,
        hidden_size=32,
        dim=32,
        hidden_dim=64,
        num_labels=3,
    )
    model = DistilBertForSequenceClassification(cfg)
    out = tmp_path / "tiny-base"
    model.save_pretrained(out)

    # Build a tokenizer from a tiny vocab.
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"tok{i}" for i in range(195)]
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tok = DistilBertTokenizerFast(vocab_file=str(vocab_path))
    tok.save_pretrained(out)
    return out


def test_load_scorer_artifacts_local_no_adapter(tmp_path: Path):
    base_dir = _make_tiny_base(tmp_path)
    # When repo_id_or_path == base_model, no PEFT adapter is applied (the loader
    # tolerates this for testing — production callers always pass an adapter).
    model, tokenizer = load_scorer_artifacts(
        repo_id_or_path=str(base_dir),
        base_model=str(base_dir),
        device="cpu",
    )
    assert hasattr(model, "forward")
    # Inference mode.
    assert not model.training
    # Tokenizer round-trips.
    enc = tokenizer("hello", "world", truncation=True, max_length=16, return_tensors="pt")
    assert "input_ids" in enc
    # Model runs forward.
    with torch.no_grad():
        out = model(**enc)
    assert out.logits.shape[-1] == 3


def test_load_scorer_artifacts_falls_back_to_base_tokenizer(tmp_path: Path):
    base_dir = _make_tiny_base(tmp_path)
    # Make a directory that has a model but no tokenizer files.
    adapter_dir = tmp_path / "adapter-no-tokenizer"
    adapter_dir.mkdir()
    # Copy just the model weights — no tokenizer.
    for name in ("config.json", "model.safetensors", "pytorch_model.bin"):
        src = base_dir / name
        if src.exists():
            (adapter_dir / name).write_bytes(src.read_bytes())

    model, tokenizer = load_scorer_artifacts(
        repo_id_or_path=str(adapter_dir),
        base_model=str(base_dir),
        device="cpu",
    )
    # Tokenizer came from the base model.
    assert tokenizer is not None
    enc = tokenizer("hello", "world", truncation=True, max_length=16, return_tensors="pt")
    assert "input_ids" in enc


@pytest.mark.integration
def test_load_scorer_artifacts_from_hub():
    """Smoke-load the published Phase 3 model.

    Requires HF_USERNAME env var and a model at <HF_USERNAME>/resumora-ai-distilbert-lora.
    Skipped by default; run with `pytest -m integration`.
    """
    import os

    repo = os.environ.get("RESUMORA_AI_SCORER_REPO")
    if not repo:
        pytest.skip("RESUMORA_AI_SCORER_REPO not set")
    model, tokenizer = load_scorer_artifacts(
        repo_id_or_path=repo,
        base_model="distilbert-base-uncased",
        device="cpu",
    )
    assert model is not None
    assert tokenizer is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest packages/pipeline/tests/scoring/test_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.scoring.loader'`.

- [ ] **Step 3: Implement `loader.py`**

Create `packages/pipeline/src/pipeline/scoring/loader.py`:

```python
from __future__ import annotations

from pathlib import Path

from peft import PeftConfig, PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _has_peft_adapter(repo_id_or_path: str) -> bool:
    """Detect a PEFT adapter at a local path. Hub lookups are best-effort."""
    p = Path(repo_id_or_path)
    if p.exists():
        return (p / "adapter_config.json").exists()
    # For Hub repos, let the PEFT loader try and surface its own error.
    try:
        PeftConfig.from_pretrained(repo_id_or_path)
        return True
    except Exception:  # noqa: BLE001
        return False


def _load_tokenizer(repo_id_or_path: str, fallback_base_model: str):
    """Try the adapter repo first; fall back to the base model's tokenizer."""
    try:
        return AutoTokenizer.from_pretrained(repo_id_or_path)
    except (OSError, ValueError):
        return AutoTokenizer.from_pretrained(fallback_base_model)


def load_scorer_artifacts(
    *,
    repo_id_or_path: str,
    base_model: str = "distilbert-base-uncased",
    device: str = "cpu",
):
    """Load the (model, tokenizer) pair for inference.

    `repo_id_or_path` may be a Hub repo (`USER/model-name`) or a local directory.
    If a PEFT adapter is detected, the base model is loaded first and the adapter
    is applied on top; otherwise `repo_id_or_path` is loaded as a standalone model
    (this path is mostly used in tests).
    """
    tokenizer = _load_tokenizer(repo_id_or_path, base_model)

    if _has_peft_adapter(repo_id_or_path):
        base = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=3)
        model = PeftModel.from_pretrained(base, repo_id_or_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            repo_id_or_path, num_labels=3
        )

    model.to(device)
    # Inference mode (equivalent to PyTorch's shorter-named method; we use this
    # spelling consistently in the codebase — see Phase 3 supplement note).
    model.train(False)
    return model, tokenizer
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest packages/pipeline/tests/scoring/test_loader.py -v`
Expected: PASS (2 tests; 1 skipped — the integration test).

- [ ] **Step 5: Commit**

```bash
git add packages/pipeline/src/pipeline/scoring/loader.py packages/pipeline/tests/scoring/test_loader.py
git commit -m "feat: add scoring artifact loader with PEFT detection + tokenizer fallback"
```

---

## Task 5: `Scorer` class

**Files:**
- Create: `packages/pipeline/src/pipeline/scoring/scorer.py`
- Modify: `packages/pipeline/src/pipeline/scoring/__init__.py`
- Create: `packages/pipeline/tests/scoring/test_scorer.py`

`Scorer` is the public surface for Phase 4 scoring. `Scorer.from_pretrained(repo_id_or_path)` eagerly loads the model and tokenizer; `scorer.score(resume_text, jd_text)` runs inference and returns a `ScoreResult`.

- [ ] **Step 1: Write the failing tests**

Create `packages/pipeline/tests/scoring/test_scorer.py`:

```python
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizerFast

from pipeline.scoring import Scorer, ScoreResult


def _build_deterministic_scorer(tmp_path: Path, bias_to_class: int) -> Scorer:
    """Save a tiny DistilBERT whose classifier is biased to a chosen class.

    Setting the classifier bias to a large value for one class guarantees that
    every input is predicted as that class — useful for asserting Scorer behavior
    without relying on the random init of a tiny model.
    """
    cfg = DistilBertConfig(
        vocab_size=200,
        max_position_embeddings=64,
        num_hidden_layers=1,
        n_layers=1,
        n_heads=2,
        hidden_size=32,
        dim=32,
        hidden_dim=64,
        num_labels=3,
    )
    model = DistilBertForSequenceClassification(cfg)
    # Zero out the classifier weights and bias it heavily to the chosen class.
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.zero_()
        model.classifier.bias[bias_to_class] = 50.0
    out = tmp_path / "tiny"
    model.save_pretrained(out)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"tok{i}" for i in range(195)]
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tok = DistilBertTokenizerFast(vocab_file=str(vocab_path))
    tok.save_pretrained(out)
    return Scorer.from_pretrained(repo_id_or_path=str(out), base_model=str(out), device="cpu")


def test_scorer_returns_score_result_with_strong_bias(tmp_path: Path):
    scorer = _build_deterministic_scorer(tmp_path, bias_to_class=2)
    result = scorer.score("alice has 5 years of python", "we need a senior python engineer")
    assert isinstance(result, ScoreResult)
    assert result.predicted_label == "strong"
    assert result.score > 80
    assert result.confidence > 0.99
    # Class probabilities sum to ~1.
    total = sum(result.class_probabilities.values())
    assert abs(total - 1.0) < 1e-5


def test_scorer_returns_score_result_with_weak_bias(tmp_path: Path):
    scorer = _build_deterministic_scorer(tmp_path, bias_to_class=0)
    result = scorer.score("alice studied english", "we need a senior python engineer")
    assert result.predicted_label == "weak"
    assert result.score < 25
    assert result.confidence > 0.99


def test_scorer_max_length_param_truncates(tmp_path: Path):
    scorer = _build_deterministic_scorer(tmp_path, bias_to_class=1)
    long_text = "tok1 " * 1000
    # Should not raise even though tokenized length far exceeds the model max.
    result = scorer.score(long_text, long_text)
    assert isinstance(result, ScoreResult)


def test_scorer_is_repeatable(tmp_path: Path):
    """No randomness — same input → same output (model is in eval mode)."""
    scorer = _build_deterministic_scorer(tmp_path, bias_to_class=2)
    r1 = scorer.score("a", "b")
    r2 = scorer.score("a", "b")
    assert r1.score == r2.score
    assert r1.class_probabilities == r2.class_probabilities


def test_scoring_public_surface():
    """Both names are re-exported from the package root."""
    from pipeline.scoring import Scorer as S, ScoreResult as R

    assert S is Scorer
    assert R is ScoreResult
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest packages/pipeline/tests/scoring/test_scorer.py -v`
Expected: FAIL with `ImportError` from `pipeline.scoring`.

- [ ] **Step 3: Implement `scorer.py`**

Create `packages/pipeline/src/pipeline/scoring/scorer.py`:

```python
from __future__ import annotations

from typing import Self

import numpy as np
import torch

from pipeline.scoring._math import (
    INT_TO_LABEL,
    confidence_from_logits,
    score_from_logits,
    softmax,
)
from pipeline.scoring.loader import load_scorer_artifacts
from pipeline.scoring.models import ScoreResult


class Scorer:
    """Eagerly-loaded inference wrapper for the Phase 3 DistilBERT+LoRA model.

    Score range is bounded to [20.0, 85.0] — see ScoreResult.score and the
    Phase 3 supplement §1.1.
    """

    def __init__(self, *, model, tokenizer, device: str, max_length: int = 512) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._max_length = max_length

    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_path: str,
        *,
        base_model: str = "distilbert-base-uncased",
        device: str = "cpu",
        max_length: int = 512,
    ) -> Self:
        """Load model + tokenizer eagerly and return a ready-to-use Scorer."""
        model, tokenizer = load_scorer_artifacts(
            repo_id_or_path=repo_id_or_path,
            base_model=base_model,
            device=device,
        )
        return cls(model=model, tokenizer=tokenizer, device=device, max_length=max_length)

    def score(self, resume_text: str, jd_text: str) -> ScoreResult:
        """Classify a (resume, JD) pair and return score + probabilities."""
        enc = self._tokenizer(
            resume_text,
            jd_text,
            truncation=True,
            max_length=self._max_length,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            out = self._model(**enc)
        logits_np = out.logits.detach().cpu().numpy()  # shape (1, 3)

        probs = softmax(logits_np)[0]
        score = float(score_from_logits(logits_np)[0])
        confidence = float(confidence_from_logits(logits_np)[0])
        pred_int = int(np.argmax(probs))

        return ScoreResult(
            score=score,
            confidence=confidence,
            class_probabilities={INT_TO_LABEL[i]: float(probs[i]) for i in range(3)},
            predicted_label=INT_TO_LABEL[pred_int],  # type: ignore[arg-type]
        )
```

- [ ] **Step 4: Re-export the public surface**

Replace `packages/pipeline/src/pipeline/scoring/__init__.py` with:

```python
"""Phase 4 scoring library: load the fine-tuned DistilBERT+LoRA model and score (resume, JD) pairs."""

from pipeline.scoring.models import PREDICTED_LABELS, PredictedLabel, ScoreResult
from pipeline.scoring.scorer import Scorer

__all__ = [
    "PREDICTED_LABELS",
    "PredictedLabel",
    "ScoreResult",
    "Scorer",
]
```

- [ ] **Step 5: Run the scorer tests to verify they pass**

Run: `uv run pytest packages/pipeline/tests/scoring/test_scorer.py -v`
Expected: PASS (5 tests).

- [ ] **Step 6: Run the full scoring test suite**

Run: `uv run pytest packages/pipeline/tests/scoring/ -v`
Expected: all PASS (1 integration test skipped).

- [ ] **Step 7: Commit**

```bash
git add packages/pipeline/src/pipeline/scoring/scorer.py packages/pipeline/src/pipeline/scoring/__init__.py packages/pipeline/tests/scoring/test_scorer.py
git commit -m "feat: add Scorer class for (resume, JD) → ScoreResult inference"
```

---

## Task 6: Similarity sub-package skeleton + models

**Files:**
- Create: `packages/pipeline/src/pipeline/similarity/__init__.py`
- Create: `packages/pipeline/src/pipeline/similarity/models.py`
- Create: `packages/pipeline/tests/similarity/__init__.py`
- Create: `packages/pipeline/tests/similarity/test_models.py`

`SkillMatch` and `SkillMatchReport` are the data shapes that the matcher returns. Both are Pydantic models, mirroring `ScoreResult` and `ResumeProfile`.

- [ ] **Step 1: Write the failing tests**

Create `packages/pipeline/tests/similarity/test_models.py`:

```python
import pytest
from pydantic import ValidationError

from pipeline.similarity.models import SkillMatch, SkillMatchReport


def test_skill_match_valid():
    m = SkillMatch(
        jd_skill="python",
        resume_skill="Python",
        similarity=0.92,
        matched=True,
    )
    assert m.jd_skill == "python"
    assert m.matched is True


def test_skill_match_similarity_out_of_range_rejected():
    with pytest.raises(ValidationError):
        SkillMatch(jd_skill="a", resume_skill="b", similarity=1.5, matched=False)


def test_skill_match_negative_similarity_rejected():
    with pytest.raises(ValidationError):
        SkillMatch(jd_skill="a", resume_skill="b", similarity=-0.1, matched=False)


def test_skill_match_resume_skill_may_be_empty_when_resume_has_no_skills():
    # Documented: if the resume has zero skills, "best resume skill" is "".
    m = SkillMatch(jd_skill="python", resume_skill="", similarity=0.0, matched=False)
    assert m.resume_skill == ""


def test_skill_match_report_match_rate():
    report = SkillMatchReport(
        required_matched=[SkillMatch(jd_skill="a", resume_skill="A", similarity=0.9, matched=True)],
        required_missing=[SkillMatch(jd_skill="b", resume_skill="x", similarity=0.1, matched=False)],
        nice_to_have_matched=[],
        nice_to_have_missing=[],
        match_rate=0.5,
    )
    assert report.match_rate == 0.5


def test_skill_match_report_match_rate_bounded():
    with pytest.raises(ValidationError):
        SkillMatchReport(
            required_matched=[],
            required_missing=[],
            nice_to_have_matched=[],
            nice_to_have_missing=[],
            match_rate=1.5,
        )
```

- [ ] **Step 2: Create the empty similarity package and tests markers**

Create `packages/pipeline/src/pipeline/similarity/__init__.py` with placeholder content (will be replaced in Task 8):

```python
"""Phase 4 similarity library: skill-level matching via sentence-transformer embeddings."""
```

Create `packages/pipeline/tests/similarity/__init__.py` as a zero-byte file.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest packages/pipeline/tests/similarity/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.similarity.models'`.

- [ ] **Step 4: Implement `models.py`**

Create `packages/pipeline/src/pipeline/similarity/models.py`:

```python
from __future__ import annotations

from pydantic import BaseModel, Field


class SkillMatch(BaseModel):
    """A single JD skill compared against the closest resume skill."""

    jd_skill: str = Field(description="The JD skill being matched.")
    resume_skill: str = Field(
        description="The closest resume skill (empty string when the resume has no skills).",
    )
    similarity: float = Field(ge=0.0, le=1.0, description="Cosine similarity in [0, 1].")
    matched: bool = Field(description="True if similarity >= matcher threshold.")


class SkillMatchReport(BaseModel):
    """Aggregate result of `SkillMatcher.match(resume, job)`."""

    required_matched: list[SkillMatch]
    required_missing: list[SkillMatch]
    nice_to_have_matched: list[SkillMatch]
    nice_to_have_missing: list[SkillMatch]
    match_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="matched-required / total-required. 1.0 when total-required == 0.",
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest packages/pipeline/tests/similarity/test_models.py -v`
Expected: PASS (6 tests).

- [ ] **Step 6: Commit**

```bash
git add packages/pipeline/src/pipeline/similarity/__init__.py packages/pipeline/src/pipeline/similarity/models.py packages/pipeline/tests/similarity/__init__.py packages/pipeline/tests/similarity/test_models.py
git commit -m "feat: add SkillMatch and SkillMatchReport pydantic models"
```

---

## Task 7: Embedding backend protocol + sentence-transformers wrapper

**Files:**
- Create: `packages/pipeline/src/pipeline/similarity/_embeddings.py`
- Create: `packages/pipeline/tests/similarity/test_embeddings.py`

`EmbeddingBackend` is a Protocol so tests can inject a fake. The default `SentenceTransformerBackend` lazily imports `sentence_transformers` so the test module never needs the real model.

- [ ] **Step 1: Write the failing tests**

Create `packages/pipeline/tests/similarity/test_embeddings.py`:

```python
import numpy as np
import pytest

from pipeline.similarity._embeddings import EmbeddingBackend, SentenceTransformerBackend


class _FakeBackend:
    """Maps each unique text to a fixed orthogonal-ish unit vector."""

    def __init__(self) -> None:
        self._seen: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for t in texts:
            if t not in self._seen:
                vec = np.zeros(8)
                # Deterministic spot per text.
                vec[hash(t) % 8] = 1.0
                self._seen[t] = vec
            rows.append(self._seen[t])
        return np.stack(rows)


def test_fake_backend_returns_unit_vectors():
    backend: EmbeddingBackend = _FakeBackend()
    out = backend.encode(["python", "java"])
    assert out.shape == (2, 8)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0])


@pytest.mark.integration
def test_sentence_transformer_backend_real_model():
    """Loads sentence-transformers/all-MiniLM-L6-v2. Gated; skipped by default."""
    backend = SentenceTransformerBackend("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    out = backend.encode(["python", "Python"])
    # Default normalization is on.
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0], atol=1e-3)
    # The two strings should be very close.
    sim = float(out[0] @ out[1])
    assert sim > 0.9
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest packages/pipeline/tests/similarity/test_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.similarity._embeddings'`.

- [ ] **Step 3: Implement `_embeddings.py`**

Create `packages/pipeline/src/pipeline/similarity/_embeddings.py`:

```python
from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingBackend(Protocol):
    """Anything that turns a list of texts into an (N, D) array of unit vectors."""

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerBackend:
    """Default backend: wraps `sentence_transformers.SentenceTransformer`.

    Imports are lazy so this module loads instantly in unit tests that inject
    a fake backend.
    """

    def __init__(self, model_name: str, *, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        # normalize_embeddings=True makes cosine reduce to a dot product, which
        # keeps the matcher arithmetic numerically tight.
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True),
            dtype=np.float64,
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest packages/pipeline/tests/similarity/test_embeddings.py -v`
Expected: PASS (1 test; 1 skipped integration).

- [ ] **Step 5: Commit**

```bash
git add packages/pipeline/src/pipeline/similarity/_embeddings.py packages/pipeline/tests/similarity/test_embeddings.py
git commit -m "feat: add EmbeddingBackend protocol and sentence-transformers wrapper"
```

---

## Task 8: `SkillMatcher` class

**Files:**
- Create: `packages/pipeline/src/pipeline/similarity/matcher.py`
- Modify: `packages/pipeline/src/pipeline/similarity/__init__.py`
- Create: `packages/pipeline/tests/similarity/test_matcher.py`

`SkillMatcher` takes an `EmbeddingBackend` (or builds the default one via `from_pretrained`) and a threshold. `matcher.match(resume_profile, job_profile)` computes the report.

- [ ] **Step 1: Write the failing tests**

Create `packages/pipeline/tests/similarity/test_matcher.py`:

```python
import numpy as np
import pytest

from pipeline.extraction.models import JobProfile, ResumeProfile
from pipeline.similarity import SkillMatcher, SkillMatchReport


class _FixedBackend:
    """Returns hand-set unit vectors for a known vocabulary."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        # Normalize for safety.
        self._vecs = {}
        for k, v in vectors.items():
            arr = np.array(v, dtype=np.float64)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            self._vecs[k] = arr

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vecs[t.lower().strip()] for t in texts])


def _resume(skills: list[str]) -> ResumeProfile:
    return ResumeProfile(
        titles=["engineer"],
        skills=skills,
        experiences=[],
        education=[],
        total_years_experience=3.0,
    )


def _job(required: list[str], nice: list[str]) -> JobProfile:
    return JobProfile(
        title="engineer",
        required_skills=required,
        nice_to_have_skills=nice,
        seniority="mid",
        min_years_experience=2.0,
    )


def test_match_above_threshold_counts_as_matched():
    backend = _FixedBackend({"python": [1, 0], "python3": [0.95, 0.3]})  # cos ≈ 0.95
    matcher = SkillMatcher(backend=backend, threshold=0.55)
    report = matcher.match(_resume(["Python3"]), _job(["Python"], []))
    assert len(report.required_matched) == 1
    assert report.required_matched[0].jd_skill == "Python"
    assert report.required_matched[0].resume_skill == "Python3"
    assert report.required_matched[0].matched is True
    assert report.required_missing == []


def test_match_below_threshold_counts_as_missing():
    backend = _FixedBackend({"python": [1, 0], "java": [0, 1]})  # cos = 0
    matcher = SkillMatcher(backend=backend, threshold=0.55)
    report = matcher.match(_resume(["Java"]), _job(["Python"], []))
    assert report.required_matched == []
    assert len(report.required_missing) == 1
    miss = report.required_missing[0]
    assert miss.jd_skill == "Python"
    assert miss.resume_skill == "Java"  # closest resume skill, even though rejected
    assert miss.matched is False


def test_match_rate_required_only():
    backend = _FixedBackend({
        "python": [1, 0, 0],
        "django": [0, 1, 0],
        "rust": [0, 0, 1],
    })
    matcher = SkillMatcher(backend=backend, threshold=0.55)
    # Two required (python, django), one matched, one not.
    report = matcher.match(_resume(["Python"]), _job(["Python", "Django"], []))
    assert report.match_rate == 0.5


def test_empty_resume_skills_means_everything_missing():
    backend = _FixedBackend({"python": [1, 0]})
    matcher = SkillMatcher(backend=backend, threshold=0.55)
    report = matcher.match(_resume([]), _job(["Python"], []))
    assert report.required_matched == []
    assert len(report.required_missing) == 1
    miss = report.required_missing[0]
    assert miss.resume_skill == ""
    assert miss.similarity == 0.0
    assert report.match_rate == 0.0


def test_empty_required_skills_match_rate_one():
    backend = _FixedBackend({"python": [1, 0]})
    matcher = SkillMatcher(backend=backend, threshold=0.55)
    report = matcher.match(_resume(["Python"]), _job([], []))
    assert report.match_rate == 1.0
    assert report.required_matched == []
    assert report.required_missing == []


def test_nice_to_have_uses_same_threshold():
    backend = _FixedBackend({"docker": [1, 0], "kubernetes": [0, 1]})
    matcher = SkillMatcher(backend=backend, threshold=0.55)
    # No required; one nice that matches, one that doesn't.
    report = matcher.match(_resume(["Docker"]), _job([], ["Docker", "Kubernetes"]))
    assert len(report.nice_to_have_matched) == 1
    assert len(report.nice_to_have_missing) == 1
    # match_rate only reflects required.
    assert report.match_rate == 1.0


def test_normalization_lowers_and_strips():
    # "PYTHON " and "python" must map to the same vector via the lower/strip step.
    backend = _FixedBackend({"python": [1, 0]})
    matcher = SkillMatcher(backend=backend, threshold=0.55)
    report = matcher.match(_resume(["PYTHON "]), _job([" python"], []))
    assert len(report.required_matched) == 1
    assert report.required_matched[0].similarity > 0.99


def test_threshold_is_inclusive_at_boundary():
    # Build vectors at exact similarity = 0.55.
    # cos = 0.55 when v1 = [1, 0], v2 = [0.55, sqrt(1 - 0.55**2)].
    import math
    a = [1.0, 0.0]
    b = [0.55, math.sqrt(1 - 0.55 ** 2)]
    backend = _FixedBackend({"a": a, "b": b})
    matcher = SkillMatcher(backend=backend, threshold=0.55)
    report = matcher.match(_resume(["A"]), _job(["B"], []))
    assert len(report.required_matched) == 1  # >= threshold counts
    assert report.required_matched[0].matched is True


def test_public_surface_exports_all_names():
    from pipeline.similarity import SkillMatch, SkillMatcher as M, SkillMatchReport as R

    assert M is SkillMatcher
    assert R is SkillMatchReport
    assert SkillMatch is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest packages/pipeline/tests/similarity/test_matcher.py -v`
Expected: FAIL with `ImportError`/`ModuleNotFoundError` from `pipeline.similarity`.

- [ ] **Step 3: Implement `matcher.py`**

Create `packages/pipeline/src/pipeline/similarity/matcher.py`:

```python
from __future__ import annotations

from typing import Self

import numpy as np

from pipeline.extraction.models import JobProfile, ResumeProfile
from pipeline.similarity._embeddings import EmbeddingBackend, SentenceTransformerBackend
from pipeline.similarity.models import SkillMatch, SkillMatchReport

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_THRESHOLD = 0.55


def _normalize(skill: str) -> str:
    return skill.lower().strip()


class SkillMatcher:
    """Embedding-based skill matcher.

    For every JD skill, the closest resume skill (by cosine similarity) is the
    candidate. If the similarity meets `threshold`, the pair lands in the
    matched list; otherwise it lands in the missing list (still carrying the
    closest-but-rejected resume skill, for downstream messaging).
    """

    def __init__(self, *, backend: EmbeddingBackend, threshold: float = _DEFAULT_THRESHOLD) -> None:
        self._backend = backend
        self._threshold = threshold

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = _DEFAULT_MODEL,
        *,
        device: str = "cpu",
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> Self:
        backend = SentenceTransformerBackend(model_name, device=device)
        return cls(backend=backend, threshold=threshold)

    def match(self, resume: ResumeProfile, job: JobProfile) -> SkillMatchReport:
        required_matched, required_missing = self._match_list(
            resume_skills=resume.skills, jd_skills=job.required_skills
        )
        nice_matched, nice_missing = self._match_list(
            resume_skills=resume.skills, jd_skills=job.nice_to_have_skills
        )
        total_required = len(job.required_skills)
        match_rate = 1.0 if total_required == 0 else len(required_matched) / total_required
        return SkillMatchReport(
            required_matched=required_matched,
            required_missing=required_missing,
            nice_to_have_matched=nice_matched,
            nice_to_have_missing=nice_missing,
            match_rate=match_rate,
        )

    def _match_list(
        self,
        *,
        resume_skills: list[str],
        jd_skills: list[str],
    ) -> tuple[list[SkillMatch], list[SkillMatch]]:
        matched: list[SkillMatch] = []
        missing: list[SkillMatch] = []
        if not jd_skills:
            return matched, missing

        if not resume_skills:
            # No resume skills → every JD skill is missing with similarity 0.
            for jd_skill in jd_skills:
                missing.append(
                    SkillMatch(jd_skill=jd_skill, resume_skill="", similarity=0.0, matched=False)
                )
            return matched, missing

        jd_norm = [_normalize(s) for s in jd_skills]
        rs_norm = [_normalize(s) for s in resume_skills]
        jd_vecs = self._backend.encode(jd_norm)
        rs_vecs = self._backend.encode(rs_norm)

        # Cosine on unit-normalized vectors is the dot product.
        sims = jd_vecs @ rs_vecs.T  # shape (J, R)

        for i, jd_skill in enumerate(jd_skills):
            row = sims[i]
            best_idx = int(np.argmax(row))
            best_sim = float(row[best_idx])
            # Clamp into [0, 1] — sentence-transformer cosine can dip very slightly negative.
            best_sim = max(0.0, min(1.0, best_sim))
            entry = SkillMatch(
                jd_skill=jd_skill,
                resume_skill=resume_skills[best_idx],
                similarity=best_sim,
                matched=best_sim >= self._threshold,
            )
            (matched if entry.matched else missing).append(entry)
        return matched, missing
```

- [ ] **Step 4: Re-export the public surface**

Replace `packages/pipeline/src/pipeline/similarity/__init__.py` with:

```python
"""Phase 4 similarity library: skill-level matching via sentence-transformer embeddings."""

from pipeline.similarity.matcher import SkillMatcher
from pipeline.similarity.models import SkillMatch, SkillMatchReport

__all__ = [
    "SkillMatch",
    "SkillMatchReport",
    "SkillMatcher",
]
```

- [ ] **Step 5: Run the matcher tests to verify they pass**

Run: `uv run pytest packages/pipeline/tests/similarity/test_matcher.py -v`
Expected: PASS (9 tests).

- [ ] **Step 6: Run the full similarity test suite**

Run: `uv run pytest packages/pipeline/tests/similarity/ -v`
Expected: all PASS (1 integration test skipped).

- [ ] **Step 7: Commit**

```bash
git add packages/pipeline/src/pipeline/similarity/matcher.py packages/pipeline/src/pipeline/similarity/__init__.py packages/pipeline/tests/similarity/test_matcher.py
git commit -m "feat: add SkillMatcher for required/nice-to-have skill matching"
```

---

## Task 9: Phase 4 docs and README entry

**Files:**
- Create: `docs/phase-4-score.md`
- Modify: `README.md`

- [ ] **Step 1: Write the Phase 4 user guide**

Create `docs/phase-4-score.md` with:

```markdown
# Phase 4 — Score & Similarity

Two pure libraries that turn a `(resume, job description)` pair into a fit score and a matched / missing skills report.

## What ships

- `pipeline.scoring.Scorer` — loads the fine-tuned Phase 3 DistilBERT+LoRA model from HF Hub (or a local directory) and scores a pair.
- `pipeline.similarity.SkillMatcher` — embeds skill phrases with `sentence-transformers/all-MiniLM-L6-v2` and returns a `SkillMatchReport`.

Neither module depends on FastAPI, Ollama, or `packages/training`. They are imported by the Phase 6 API.

## Score range is `[20, 85]`

The scorer's output is bounded to `[20.0, 85.0]` — softmax · `[20, 55, 85]`. This is a deliberate Phase 3 design decision (see `docs/superpowers/specs/2026-05-15-phase-3-finetune-supplement.md` §1.1). The product surface ("0–100 fit score") is honored by disclosure, not by rescaling.

## Usage

```python
from pipeline.scoring import Scorer
from pipeline.similarity import SkillMatcher
from pipeline.extraction import extract_resume_profile, extract_job_profile
from pipeline.ingestion import ingest_resume_bytes, ingest_job_text

scorer = Scorer.from_pretrained("USER/resumora-ai-distilbert-lora", device="cpu")
matcher = SkillMatcher.from_pretrained(device="cpu")  # MiniLM by default

resume_doc = ingest_resume_bytes(resume_pdf_bytes, filename="resume.pdf")
job_doc = ingest_job_text(jd_text)

resume_profile = extract_resume_profile(resume_doc)
job_profile = extract_job_profile(job_doc)

result = scorer.score(resume_doc.raw_text, job_doc.raw_text)
report = matcher.match(resume_profile, job_profile)

print(f"Score: {result.score:.1f} (confidence {result.confidence:.2f})")
print(f"Required match rate: {report.match_rate:.0%}")
for miss in report.required_missing:
    print(f"  missing: {miss.jd_skill}  (closest: {miss.resume_skill}, {miss.similarity:.2f})")
```

## Loading from a local adapter

For development against a smoke-trained adapter:

```python
scorer = Scorer.from_pretrained(
    repo_id_or_path="outputs/smoke",   # local PEFT adapter dir
    base_model="distilbert-base-uncased",
    device="cpu",
)
```

## Tuning the similarity threshold

The default threshold is `0.55` (cosine on MiniLM normalized embeddings). Lower to admit more matches; raise to be stricter:

```python
matcher = SkillMatcher.from_pretrained(threshold=0.6)
```

## Testing

Unit tests do not download models: scoring uses a config-built tiny DistilBERT; similarity uses an injected fake `EmbeddingBackend`. Integration tests (real Hub model, real sentence-transformer) are gated:

```bash
# unit tests (default)
uv run pytest packages/pipeline/tests/scoring packages/pipeline/tests/similarity -v

# integration tests
uv run pytest packages/pipeline/tests/scoring packages/pipeline/tests/similarity -m integration -v
```

The Hub integration test needs `RESUMORA_AI_SCORER_REPO` pointing at the published Phase 3 model:

```bash
RESUMORA_AI_SCORER_REPO=USER/resumora-ai-distilbert-lora \
  uv run pytest packages/pipeline/tests/scoring -m integration -v
```
```

- [ ] **Step 2: Add the Phase 4 entry to the README phases list**

In `README.md`, replace this line:

```markdown
- **Phase 3 — fine-tune:** [plan](docs/superpowers/plans/2026-05-15-phase-3-finetune.md), [supplement](docs/superpowers/specs/2026-05-15-phase-3-finetune-supplement.md), [guide](docs/phase-3-finetune.md).
```

with:

```markdown
- **Phase 3 — fine-tune:** [plan](docs/superpowers/plans/2026-05-15-phase-3-finetune.md), [supplement](docs/superpowers/specs/2026-05-15-phase-3-finetune-supplement.md), [guide](docs/phase-3-finetune.md).
- **Phase 4 — score & similarity:** [plan](docs/superpowers/plans/2026-05-15-phase-4-score.md), [supplement](docs/superpowers/specs/2026-05-15-phase-4-score-supplement.md), [guide](docs/phase-4-score.md).
```

- [ ] **Step 3: Run the entire test suite to ensure nothing else regressed**

Run: `uv run pytest -v`
Expected: all tests pass (existing ingestion/extraction tests + new Phase 4 tests); integration tests skipped by default.

- [ ] **Step 4: Commit**

```bash
git add docs/phase-4-score.md README.md
git commit -m "docs: add Phase 4 user guide and README entry"
```

---

## Done check

- [ ] All Phase 4 tasks committed.
- [ ] `uv run pytest -v` passes with no failures.
- [ ] `pipeline.scoring.Scorer` and `pipeline.similarity.SkillMatcher` are importable from the package roots.
- [ ] The score range `[20, 85]` is documented in: the Scorer docstring, the `ScoreResult.score` field description, and `docs/phase-4-score.md`.
- [ ] No `pipeline → training` imports were introduced (`grep -R "from training" packages/pipeline/src` returns nothing).
