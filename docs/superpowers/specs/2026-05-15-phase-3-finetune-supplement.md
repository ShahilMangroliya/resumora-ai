# Phase 3 — Fine-tune: design supplement

**Date:** 2026-05-15
**Status:** Approved
**Supplements:** [`2026-05-14-ai-pipeline-design.md`](2026-05-14-ai-pipeline-design.md) — does not replace it.

This document locks in the Phase 3 decisions that the master spec deliberately left open: task formulation, base model, LoRA config, compute split, evaluation, and how the trained model is published. Wherever this supplement and the master spec disagree, the supplement wins for Phase 3 only.

## 1. Task formulation

**3-class classification with expected-value scoring at inference.**

The Phase 2 dataset uses three labels (`weak`, `partial`, `strong`) with fixed bucket scores (`20`, `55`, `85`) defined in `training.dataset.schema.LABEL_TO_SCORE`. The model is trained as a standard 3-class classifier with cross-entropy loss. At inference, the score is the softmax-weighted average of the three bucket scores:

```
score = softmax(logits) · [20, 55, 85]
confidence = max(softmax(logits))
```

This matches the data exactly (three distinct labels), produces a smooth continuous output between bucket midpoints, and gives a per-class probability that doubles as a confidence signal.

### 1.1 Score range is [20, 85] — disclose this everywhere

A softmax-weighted average of `[20, 55, 85]` can never produce a value below 20 or above 85. The master spec describes a "0–100 score" — that is the **product surface**; the **actual numeric range** under this formulation is `[20, 85]`.

This is a deliberate choice — not a bug — to keep the score honestly anchored to the bucket structure the data actually represents. Rescaling to `[0, 100]` would manufacture precision the data does not have.

**Required disclosures:**

- The Phase 3 implementation plan must include this fact in a comment on the inference function.
- The model card (§6) must include a "Score range" subsection saying scores are bounded to `[20, 85]` and why.
- The Phase 7 frontend must label the score dial accordingly (e.g., a 20 means "weak fit," not "0% match").
- The Phase 6 API response schema must document the range.

## 2. Base model and tokenization

- **Base model:** `distilbert-base-uncased` (66M params). Matches "DistilBERT-class" in the master spec; fits T4 with headroom; canonical HF tutorial path.
- **Input encoding:** `tokenizer(resume_text, jd_text, truncation=True, max_length=512, padding=False)`. Standard sentence-pair encoding with the `[SEP]` token between resume and JD. Padding is dynamic via the `DataCollatorWithPadding`.
- **Truncation behavior:** when the combined length exceeds 512 tokens, the tokenizer truncates the longer of the two. This is acceptable because the high-signal content of both documents tends to be near the top.

## 3. LoRA configuration

```python
LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_lin", "v_lin"],          # DistilBERT attention projections
    modules_to_save=["pre_classifier", "classifier"],
    bias="none",
)
```

### 3.1 `modules_to_save` is load-bearing

DistilBERT-for-sequence-classification adds two randomly-initialized heads on top of the base encoder: `pre_classifier` (a dense layer) and `classifier` (the output projection). Both start at random init and must be **fully trainable**, not LoRA-adapted.

`task_type=SEQ_CLS` causes PEFT to auto-include `classifier` in `modules_to_save`, but `pre_classifier` is historically missed. Without it, half the classification head stays at random init throughout training and the model silently underperforms.

**Plan-level guard:** the implementation plan must include a smoke-test assertion that both `pre_classifier` and `classifier` parameters have non-trivial gradients after the first training step. The assertion lives in the training smoke test, not just as a comment.

## 4. Compute split — same code, two environments

The training code lives in `packages/training/src/training/train/` as a normal Python module runnable both locally (CPU, smoke test) and on Colab (T4 GPU, full run). The Colab notebook is **thin**: clone the repo, `uv sync`, call the same training CLI entry point with a bigger config.

| Environment | Pairs | Epochs | Purpose |
|---|---|---|---|
| Local (Mac CPU) | ~30 (subset) | 1 | Smoke test: the pipeline boots, the head receives gradients, MLflow logs a run. Not a meaningful training run. |
| Colab (T4 GPU) | full synthetic set (~500–1000) | 3–5 | The real training run. The artifact is what gets published to HF Hub. |

Both runs are entered through the same `python -m training.train` CLI with different config flags. There is no Colab-specific code path beyond the notebook itself.

## 5. MLflow scope — local artifacts, no server

- MLflow logs to `./mlruns/` (added to `.gitignore`).
- No tracking server, no remote backend — this would be paid infra for no portfolio gain.
- Each training run records: hyperparameters (config dump), per-epoch train/val loss, per-epoch macro-F1, a final eval report against the gold set, and the saved adapter weights as an artifact.
- The `mlflow ui` command can browse runs locally.

### 5.1 Colab MLflow persistence

A Colab runtime's `./mlruns` is **ephemeral** — it disappears when the kernel dies. The Colab notebook must do one of:

- **(default)** zip `./mlruns/` at the end of the notebook and download it through `google.colab.files.download`. The plan task for the notebook makes this explicit.
- (optional, documented but not required) mount Google Drive and write `./mlruns/` to a Drive path.

## 6. Evaluation

### 6.1 Gold set is currently 5 pairs — explicit prerequisite gate

The master spec §6 calls for 40–60 hand-curated gold pairs. The Phase 2 deliverable seeded 5 pairs and noted that the user grows the set during Phase 3 prep.

**Phase 3 has two evaluation milestones:**

1. **Development eval (n=5):** during Phase 3 implementation, gold metrics are reported against the 5-pair seed. These numbers are **not publishable** — with n=5, macro-F1 swings several percentage points run-to-run. They exist to catch gross regressions.
2. **Publication eval (n ≥ 30):** before the trained model and its model card are pushed to the public HF Hub, the gold set is grown to at least 30 pairs by hand. This is a **prerequisite task** in the Phase 3 plan, not an afterthought.

The model card reports the n=30+ numbers, not the n=5 numbers.

### 6.2 Metrics

Every eval run reports, against the gold set:

- accuracy
- macro-F1
- per-class precision, recall, F1 (weak / partial / strong)
- mean absolute error (MAE) of the expected-value score against `LABEL_TO_SCORE[gold_label]`
- a printed 3×3 confusion matrix

All metrics are logged to MLflow as a single eval report attached to the run. The eval is a separate entry point (`python -m training.train eval ...`) so it can be run against a checkpoint without re-training.

### 6.3 Class-balance precheck

Before each training run, a one-line stats assertion checks that the synthetic training pairs are within 60/20/20 of perfect balance across the three labels. A skewed dataset silently biases the classifier and the assertion forces it into the foreground.

## 7. Model publication

A new `training.publish.model` module pushes the trained adapter and the auto-generated model card to HF Hub. It mirrors the structure of the existing dataset publisher (`training.publish.to_hf`) but builds on `huggingface_hub.ModelCard` rather than dataset-upload primitives.

**Default repo name:** `<HF_USERNAME>/resumefit-distilbert-lora`.

**License:** `apache-2.0` — matches the DistilBERT base.

### 7.1 Model card required sections

The card is the public face of the artifact. It must include:

- **Intended use:** "portfolio demonstration; **not** for hiring decisions."
- **Score range:** the `[20, 85]` disclosure from §1.1.
- **Training data:** synthetic, generated by Ollama (`llama3.2:3b`); ~N pairs across 15 roles; link to the Phase 2 dataset repo.
- **Evaluation:** metrics against the (n ≥ 30) gold set, with the confusion matrix.
- **Known limitations:** the classifier may have learned the generator's tics; gold set is small; English only; no demographic-bias evaluation.
- **Reproducibility:** training config dump, seed, base model commit hash.

## 8. Module layout for Phase 3

```
packages/training/src/training/
├── train/                            ← NEW sub-package
│   ├── __init__.py
│   ├── config.py                     TrainConfig dataclass
│   ├── data.py                       load + tokenize pairs; train/val split; balance precheck
│   ├── model.py                      build_model() — DistilBERT + LoRA (with modules_to_save)
│   ├── metrics.py                    compute_metrics() + expected-value score helper
│   ├── runner.py                     train(config) — HF Trainer wrapper, MLflow logging
│   ├── evaluate.py                   evaluate(model, gold_pairs) — gold report
│   └── cli.py                        `python -m training.train` (train / eval subcommands)
├── publish/
│   ├── to_hf.py                      (existing — dataset publisher)
│   └── model.py                      NEW — push adapter + model card to HF Hub

notebooks/                            ← NEW top-level dir
└── 01_train_on_colab.ipynb           thin: clone repo, uv sync, run training.train.cli, zip mlruns, push model
```

## 9. Out of Phase 3 scope (deliberately)

- **No `pipeline/scoring` module yet** — Phase 4 builds the inference wrapper.
- **No hyperparameter sweep** — single config; iterate only if gold metrics are bad.
- **No multi-seed runs** — one seeded run for the portfolio.
- **No bias / fairness evaluation** — listed as a model-card limitation, not a Phase 3 deliverable.

## 10. Prerequisites tracked in the plan

The Phase 3 implementation plan opens with two explicit prerequisite tasks that block training:

1. The user has run the Phase 2 synthetic generator to produce ≥ 500 training pairs and pushed them to HF Hub.
2. The gold set has grown to ≥ 30 pairs *before* the model is published (development can proceed with the 5-pair seed; publication cannot).
