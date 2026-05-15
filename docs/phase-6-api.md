# Phase 6 — API

A FastAPI app that wires the five pipeline stages into a single synchronous request.

## Endpoints

### `GET /health`

Liveness check.

```json
{"status": "ok"}
```

### `POST /analyze`

Score a resume against a job description and explain the fit.

**Request — `multipart/form-data`:**

| Field | Type | Notes |
|---|---|---|
| `resume` | file (`.pdf` / `.docx` / `.txt`) | Required. |
| `job_description` | text | Required. The JD body. |

**Response — `AnalyzeResponse` (`application/json`):**

```json
{
  "score": {
    "score": 72.5,
    "confidence": 0.84,
    "class_probabilities": {"weak": 0.05, "partial": 0.11, "strong": 0.84},
    "predicted_label": "strong"
  },
  "skill_report": {
    "required_matched": [{"jd_skill": "python", "resume_skill": "python", "similarity": 0.99, "matched": true}],
    "required_missing": [{"jd_skill": "kubernetes", "resume_skill": "docker", "similarity": 0.4, "matched": false}],
    "nice_to_have_matched": [],
    "nice_to_have_missing": [],
    "match_rate": 0.67
  },
  "reasoning": {
    "reasons": [
      {"summary": "...", "evidence": "...", "category": "matched_skill"}
    ],
    "rewrites": [
      {"original": "...", "rewritten": "...", "rationale": "..."}
    ]
  },
  "warnings": []
}
```

### Status codes

| Status | Meaning |
|---|---|
| 200 | Full result (`warnings == []`) or partial result (`warnings` explains skipped stages). |
| 400 | Resume file or JD text could not be parsed (`IngestionError`). |
| 422 | Required form field missing (FastAPI validation). |
| 500 | Scorer or another pure-Python stage crashed unexpectedly. |

### Partial-result rules

- Ingestion fails → 400.
- Extraction fails (Ollama down) → 200 with `score`, `skill_report=null`, `reasoning=null`, one warning.
- Reasoning fails (transient Ollama error or invalid model output) → 200 with `score`, `skill_report`, `reasoning=null`, one warning.
- Scoring always runs (independent of Ollama).

## Configuration

All env vars are prefixed `RESUMEFIT_`.

| Var | Default | Notes |
|---|---|---|
| `RESUMEFIT_SCORER_REPO` | `distilbert-base-uncased` | HF Hub repo for the fine-tuned scorer. Point this at your Phase 3 model. |
| `RESUMEFIT_SCORER_DEVICE` | `cpu` | `cpu` / `cuda` / `mps`. |
| `RESUMEFIT_MATCHER_DEVICE` | `cpu` | Sentence-transformer device. |
| `RESUMEFIT_OLLAMA_URL` | `http://localhost:11434` | Where Ollama is reachable. |
| `RESUMEFIT_OLLAMA_MODEL` | `llama3.2:3b` | Tag for extraction + reasoning. |
| `RESUMEFIT_OLLAMA_TIMEOUT` | `30.0` | Per-call timeout (seconds). The library default is 60s; the API tightens this for better UX. |
| `RESUMEFIT_CORS_ORIGINS` | `http://localhost:3000` | Comma-separated origins for CORS. |
| `RESUMEFIT_WARMUP_ON_STARTUP` | `false` | Production: set to `true` so model load failure surfaces at startup. Tests leave this false. |

## Timeouts

- Extraction and reasoning inherit `OllamaClient`'s configured timeout (default 30s in the API).
- Scoring, similarity, and ingestion are local CPU and have no explicit timeout.
- Worst-case wall time when Ollama hangs: `2 × extraction + 1 × reasoning` ≈ 90 s. To tighten further, lower `RESUMEFIT_OLLAMA_TIMEOUT`.

## Run locally

```bash
make dev    # starts api on :8000 and web on :3000 together
```

Or API only:

```bash
uv run --package api uvicorn api.main:app --reload --port 8000
```

For a deployed-style boot (warm singletons at startup):

```bash
RESUMEFIT_WARMUP_ON_STARTUP=true \
RESUMEFIT_SCORER_REPO=USER/resumefit-distilbert-lora \
uv run --package api uvicorn api.main:app --port 8000
```

## Testing

```bash
uv run pytest apps/api -v
```

All tests use injected fakes — no real model download and no live Ollama required.
