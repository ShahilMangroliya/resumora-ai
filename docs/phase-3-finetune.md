# Phase 3 — Fine-tune

This phase produces the trained `resumefit-distilbert-lora` model. Two paths:

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
HF_DATASET_REPO = f"{HF_USER}/resumefit-dataset"
HF_MODEL_REPO = f"{HF_USER}/resumefit-distilbert-lora"
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

Default repo: `<HF_USER>/resumefit-distilbert-lora`. The auto-generated model card includes the score-range disclosure, intended-use disclaimer ("not for hiring decisions"), synthetic-data provenance, gold-set metrics, limitations, and the training config. The license is `apache-2.0` (matches DistilBERT base).
