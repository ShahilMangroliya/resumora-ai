# Phase 4 — Score & Similarity: design supplement

**Date:** 2026-05-15
**Status:** Approved
**Supplements:** [`2026-05-14-ai-pipeline-design.md`](2026-05-14-ai-pipeline-design.md) — does not replace it.

This document locks in the Phase 4 decisions that the master spec left open: scoring API surface, model loading strategy, similarity matching strategy, embedding backend, and how Phase 4 stays independent from `packages/training`. Wherever this supplement and the master spec disagree, the supplement wins for Phase 4 only.

## 1. Scope

Phase 4 ships **two pure libraries** inside `packages/pipeline`:

- `pipeline.scoring` — loads the Phase 3 DistilBERT+LoRA adapter from HF Hub (or a local directory) and turns a `(resume_text, jd_text)` pair into a `ScoreResult` (score, confidence, per-class probabilities).
- `pipeline.similarity` — embeds each skill phrase with sentence-transformers and produces a `SkillMatchReport` of matched / missing skills for both required and nice-to-have JD skills.

Both modules are pure libraries with no FastAPI, no Ollama, no orchestration knowledge. They are imported by the Phase 6 API.

## 2. `pipeline.scoring`

### 2.1 Public surface

```python
from pipeline.scoring import Scorer, ScoreResult

scorer = Scorer.from_pretrained(
    "USER/resumefit-distilbert-lora",   # HF Hub repo id OR local path
    base_model="distilbert-base-uncased",  # default
    device="cpu",                          # default
)

result: ScoreResult = scorer.score(resume_text, jd_text)
result.score                # float in [20.0, 85.0]
result.confidence           # float in [0.0, 1.0]
result.class_probabilities  # {"weak": float, "partial": float, "strong": float}
result.predicted_label      # "weak" | "partial" | "strong"
```

`ScoreResult` is a Pydantic model (consistent with `ResumeProfile` / `JobProfile`). `predicted_label` is the argmax label.

### 2.2 Score range is `[20, 85]` — inherits from Phase 3

The same disclosure applies as in `2026-05-15-phase-3-finetune-supplement.md` §1.1:
- `Scorer.score` docstring states the range bound.
- `ScoreResult.score` field description states the range bound.
- The Phase 6 API response schema must echo this, and the Phase 7 frontend must label the dial accordingly.

### 2.3 Eager loading; fail fast

`Scorer.from_pretrained` loads weights immediately. If the model cannot be loaded, it raises before returning. The master spec §5 already specifies "HF model load fails on startup → app fails fast"; the scorer enforces this at the library boundary so the API does not have to.

### 2.4 PEFT adapter loading mirrors Phase 3 evaluate

`packages/training/src/training/train/cli.py` shows the working load pattern for a PEFT adapter (commit `adc6521`). Phase 4 re-implements that pattern in `pipeline/scoring/loader.py` — it does **not** import from `training`. The duplication is one function (~25 lines); the architectural payoff is that `pipeline` has no `training` dependency at runtime.

### 2.5 No `pipeline → training` dependency

`pipeline.scoring` re-defines its own constants:

- `INT_TO_LABEL = {0: "weak", 1: "partial", 2: "strong"}`
- `BUCKET_SCORES = [20.0, 55.0, 85.0]`
- a small `softmax` + `score_from_logits` + `confidence_from_logits` in `pipeline/scoring/_math.py`

These mirror values in `training.dataset.schema` and `training.train.metrics`. Drift risk is low because both sets are derived from the same Phase 3 supplement §1; a unit test in Phase 4 pins the values explicitly.

### 2.6 Device handling

`device` accepts `"cpu"`, `"cuda"`, or `"mps"`. The default is `"cpu"` because the API will run on HF Spaces (CPU). The Scorer calls `.to(device)` on the model once at load time; subsequent inference uses that device.

### 2.7 `model.train(False)`, not the shorter-named method

Same as Phase 3: scorer code calls `model.train(False)` to enter inference mode, never the shorter-named method. (Identical behavior in PyTorch; avoids a token that a security linter could confuse with Python's builtin code-execution function.)

## 3. `pipeline.similarity`

### 3.1 Public surface

```python
from pipeline.similarity import SkillMatcher, SkillMatchReport, SkillMatch

matcher = SkillMatcher.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",  # default
    device="cpu",                              # default
    threshold=0.55,                            # default
)

report: SkillMatchReport = matcher.match(resume_profile, job_profile)
report.required_matched       # list[SkillMatch]
report.required_missing       # list[SkillMatch]
report.nice_to_have_matched   # list[SkillMatch]
report.nice_to_have_missing   # list[SkillMatch]
report.match_rate             # matched-required / total-required, in [0.0, 1.0]
```

### 3.2 Matching strategy

For every JD skill (required and nice-to-have):

1. Embed the JD skill phrase.
2. Embed every resume skill phrase (cached per `match()` call).
3. The JD skill's best match is the resume skill with the highest cosine similarity.
4. If `best_similarity >= threshold` → `SkillMatch(jd_skill, resume_skill, similarity)` goes into the matched list. Otherwise → into the missing list.

Each `SkillMatch` carries the JD skill, the best resume skill it matched against (always populated — for misses, it is the closest-but-rejected resume skill), the similarity score, and a `matched: bool`.

### 3.3 Threshold default = `0.55`

For MiniLM-L6 cosine similarity on short skill phrases:
- 0.45–0.5 admits a lot of weak associations ("python" ↔ "scripting").
- 0.6+ rejects legitimate variants ("nodejs" ↔ "node.js").
- 0.55 is a defensible middle. Documented as tunable; the constructor accepts an override.

### 3.4 Empty edge cases

- Empty `resume_profile.skills` → every JD skill is missing; `match_rate = 0.0`.
- Empty `job_profile.required_skills` → `match_rate = 1.0` (vacuously true, documented; alternative `NaN` would force every caller to special-case it).
- Empty both → `match_rate = 1.0` with empty lists everywhere.

### 3.5 Embedding backend protocol

`pipeline/similarity/_embeddings.py` defines an `EmbeddingBackend` Protocol:

```python
class EmbeddingBackend(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...   # shape (N, D), L2-normalized
```

The default implementation wraps `sentence_transformers.SentenceTransformer` with `normalize_embeddings=True` so that cosine reduces to a dot product. Tests inject a fake backend so unit tests never download a model.

### 3.6 Case + whitespace normalization

Skill phrases are lower-cased and stripped before embedding. This is cheap and resolves "Python" vs "python" without needing the embedder to. Punctuation is not stripped — `node.js` should remain `node.js`.

### 3.7 Independence from extraction

`SkillMatcher.match` takes `ResumeProfile` and `JobProfile` from `pipeline.extraction.models`. It does not import the Ollama client or the extraction logic — it only reads the `.skills`, `.required_skills`, `.nice_to_have_skills` fields. This keeps similarity testable in isolation.

## 4. Module layout

```
packages/pipeline/src/pipeline/
├── scoring/                          ← NEW
│   ├── __init__.py                   public surface: Scorer, ScoreResult
│   ├── models.py                     ScoreResult, PREDICTED_LABELS constant
│   ├── _math.py                      softmax, score_from_logits, confidence_from_logits
│   ├── loader.py                     load_scorer_artifacts() — base + adapter from Hub/local
│   └── scorer.py                     Scorer class
└── similarity/                       ← NEW
    ├── __init__.py                   public surface: SkillMatcher, SkillMatchReport, SkillMatch
    ├── models.py                     SkillMatch, SkillMatchReport
    ├── _embeddings.py                EmbeddingBackend protocol + SentenceTransformerBackend
    └── matcher.py                    SkillMatcher class
```

Tests mirror this layout under `packages/pipeline/tests/scoring/` and `packages/pipeline/tests/similarity/`.

## 5. Dependencies added

`packages/pipeline/pyproject.toml` gains:

- `transformers>=4.44` (runtime model load for scoring)
- `torch>=2.4` (already an indirect dep via training; pinned for pipeline directly)
- `peft>=0.13` (LoRA adapter load)
- `sentence-transformers>=3.0` (similarity embeddings)

These are heavy. They are correct here because the Phase 6 API depends on `pipeline` and must run inference. No CI dependency-bloat mitigation is part of Phase 4 — the Phase 0 CI already uses `uv` caching.

## 6. Testing strategy

- **Unit tests do not download models.** For `scoring`, a config-built tiny DistilBERT (`transformers.DistilBertConfig(num_hidden_layers=1, hidden_size=32, num_labels=3)`) replaces the real model — same pattern Phase 3 uses. For `similarity`, a fake `EmbeddingBackend` returns deterministic vectors. This keeps the unit suite hermetic and CI-fast.
- **One integration test per module, gated** by `@pytest.mark.integration` (consistent with Phase 2/3): `test_scorer_loads_from_hub` and `test_skill_matcher_default_backend`. Skipped by default; run via `pytest -m integration`.
- **`_math.py` is the high-value test target** for scoring — verifies the `[20, 85]` range and the bucket-score constants.
- **Threshold semantics** for similarity are tested with deterministic vectors at exact distances.

## 7. Out of Phase 4 scope

- **No API wiring** — that is Phase 6.
- **No reasoning** — Phase 5.
- **No model versioning UI** or auto-update of the scorer — manual repo-id config only.
- **No multi-skill grouping** (e.g., "Python + Django + Flask" → "Python frameworks") — flat 1-to-1 matching is what Phase 4 ships.

## 8. Prerequisites tracked in the plan

The Phase 4 plan opens with one explicit prerequisite:

1. A Phase 3 model has been published to HF Hub at `<HF_USERNAME>/resumefit-distilbert-lora` *or* a local trained adapter directory exists. Phase 4 development can proceed against the local smoke-trained adapter (CPU run); the integration test against HF Hub is the only task that requires the published model.
