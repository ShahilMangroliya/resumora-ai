# Phase 2 — Data layer

This phase produces three artifacts that Phase 3 (fine-tuning) consumes:

1. The **extraction module** (`pipeline.extraction`) turns a `ResumeDoc`/`JobDoc` into a structured `ResumeProfile`/`JobProfile` using a local Ollama LLM.
2. The **synthetic generator** (`training.dataset`) creates `(resume, JD, label)` pairs covering 15 roles × seniorities × three fit-levels.
3. The **gold seed** (`data/gold/seed.jsonl`) holds five hand-written pairs that the model is NEVER trained on. The user grows this set during Phase 3 prep.

## Prerequisites

- Ollama running locally with `llama3.2:3b` pulled (see `docs/ollama-setup.md` from Phase 0).
- `uv sync --all-packages` from the repo root.

## Generate synthetic pairs

```bash
# Smoke test with 30 pairs (good for verifying the pipeline locally).
uv run python -m training.dataset.cli generate \
    --out data/synthetic/pairs.jsonl \
    --target 30
```

The generator is **resumable** — if you interrupt it (Ctrl-C) and re-run with a larger `--target`, it will skip pair_ids already on disk. Each pair is flushed to disk as it is produced.

Scale up to ~500–1000 once the local smoke test looks reasonable.

## Inspect dataset shape

```python
from pathlib import Path
from training.dataset.jsonl import read_pairs
from training.dataset.validate import validate

pairs = read_pairs(Path("data/synthetic/pairs.jsonl"))
report = validate(pairs)
print(report)
```

Watch label_counts, unique_roles, and duplicate counts. If a single role or label dominates, the classifier in Phase 3 will overfit.

## Run extraction against a real document

```python
from pipeline.ingestion import ingest_job
from pipeline.extraction import extract_job_profile

doc = ingest_job("…paste a real JD here…")
profile = extract_job_profile(doc)
print(profile)
```

Live-Ollama integration tests live in `packages/pipeline/tests/extraction/test_integration.py`. Run them only when Ollama is up:

```bash
uv run pytest -m integration
```

## Gold eval set

Five committed pairs in `data/gold/seed.jsonl`. The schema is the same as synthetic pairs except `source: "gold"`. Add pairs by hand — never auto-generate gold.

## USER STEP — publish to HF Hub

This step is **not** automated. It uses your personal `HF_TOKEN`:

```bash
export HF_TOKEN=hf_xxxxx  # from https://huggingface.co/settings/tokens
uv run python -m training.publish.to_hf \
    --repo <your-username>/resumefit-dataset \
    --folder data \
    --message "phase 2 — initial dataset"
```

The dataset is **public** by default. Adjust the `private` flag in `training.publish.to_hf` if you want it private during development.

## Honesty in the model card (Phase 3)

When Phase 3 publishes the trained model, the model card MUST disclose:

- Synthetic data provenance (Ollama + `llama3.2:3b`).
- Gold set size (small).
- Known risks (the classifier may learn the generator's tics; validate against gold).

Synthetic data generation is itself a skill — frame it as a technique, not a shortcut.
