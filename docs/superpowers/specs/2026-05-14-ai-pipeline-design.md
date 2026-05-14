# ResumeFit — AI Pipeline Design

**Date:** 2026-05-14
**Status:** Approved (brainstorming complete)
**Purpose:** Portfolio + learning project for an in-demand, fully open-source AI pipeline.

## 1. Overview

ResumeFit is an AI pipeline that takes a `(resume, job description)` pair and returns:

- a **fit score** from 0–100
- the **top 3 reasons** for that score (matched / missing skills, experience gaps)
- **3 specific bullet-point rewrites** to improve the resume for that job

That output is the anchor for the whole system — every component exists to produce it.

### Approach: HF fine-tuning showcase

Two tracks, each earning its place:

- **Track 1 — fine-tuned model:** a small classifier (DistilBERT-class) fine-tuned with LoRA
  on resume–JD fit, published to a public Hugging Face Hub profile with a model card. This is
  the concrete proof of model-training skill.
- **Track 2 — reasoning layer:** an open LLM (via Ollama, local) that generates the
  human-readable reasons and bullet rewrites.

The fine-tuned model produces the *score*; the LLM *explains and improves*.

RAG and agents are deliberately **out of core scope** — they would be decoration for this use
case. They are listed as clearly-labeled stretch goals only.

### Constraints

- **Fully open-source and free** — no paid APIs.
- **Compute:** fine-tuning runs on free Colab/Kaggle GPUs (T4). This forces small base models,
  LoRA/QLoRA, checkpointing to HF Hub, and runs that fit in a single ~4–6 hr session.
- **Builder profile:** comfortable with Python, new to ML/AI — sequencing favors progressive
  learning and an early working artifact.

## 2. Stack

| Layer | Choice | Why |
|---|---|---|
| Env / deps | Python 3.12 + `uv` (workspace) | fast, modern, single lockfile |
| Fine-tuning | HF Transformers + PEFT (LoRA) + Datasets, on free Colab T4 | the HF skill showcase |
| Embeddings | `sentence-transformers` (MiniLM) | runs on Mac CPU |
| Reasoning LLM | open model via Ollama (Llama 3.2 3B / Qwen2.5) | free, local, Apple Silicon |
| Backend | FastAPI | standard, marketable |
| Frontend | Next.js | full-stack web UI |
| Experiment tracking | MLflow (local) | open-source, no account needed |
| Model / dataset hosting | Hugging Face Hub | free public repos |
| Deployment | HF Spaces (api) + Vercel free tier (web) | free live demo |
| CI | GitHub Actions | free for public repos |
| Dev orchestration | Makefile + `mprocs`/`honcho` | one command surface, Python-native |

## 3. Repository structure

A clean `apps/` + `packages/` monorepo, but **without Turborepo** — the repo is ~80% Python
with a single Next.js app, so Turborepo's task-graph caching would add a JS tooling layer with
no payoff. A `uv` workspace orchestrates Python; the Next.js app is self-contained.

```
ai-pipeline/
├── Makefile                  single command surface: make dev / test / lint / build / deploy
├── pyproject.toml            uv workspace root
├── apps/
│   ├── web/                  Next.js frontend (own npm/pnpm, self-contained)
│   └── api/                  FastAPI — native Python, no package.json seam
├── packages/
│   ├── pipeline/             Python lib: ingestion, extraction, scoring, similarity, reasoning
│   └── training/             Colab notebooks + training scripts + eval harness
├── data/                     gold eval set + synthetic generation outputs
└── docs/superpowers/specs/
```

`apps/api`, `packages/pipeline`, and `packages/training` form the `uv` workspace.
`make dev` uses `mprocs` (or `honcho`) to boot web + api together.

## 4. Components

Six focused units. Stages 1–5 are pure, testable Python libraries with no web/server
knowledge; the API is the only orchestrator; the frontend only knows the API.

1. **Ingestion** (`pipeline/ingestion`) — parse resume (PDF/DOCX/text) and JD into normalized
   `ResumeDoc` / `JobDoc` objects. Deps: `pypdf`, `python-docx`.
2. **Feature extraction** (`pipeline/extraction`) — pull structured fields (skills, years of
   experience, education, titles) using the Ollama LLM. Output: `ResumeProfile` / `JobProfile`.
3. **Scoring model** (`pipeline/scoring`) — the fine-tuned DistilBERT+LoRA classifier, loaded
   from HF Hub. Input: `(resume, JD)` text pair → fit score 0–100 + confidence. Also holds the
   training entry points and eval harness (training itself runs in `packages/training`).
4. **Embedding similarity** (`pipeline/similarity`) — sentence-transformers embeddings of
   resume vs JD → matched / missing skills map per section.
5. **Reasoning layer** (`pipeline/reasoning`) — Ollama LLM; takes score + skill-match map +
   profiles → top-3 reasons and 3 bullet rewrites. Swappable model, prompt-engineering module.
6. **API + orchestration** (`apps/api`) — FastAPI app wiring stages 1→5 into one `/analyze`
   endpoint. Owns orchestration; everything else stays a pure library.

Plus **`apps/web`** — Next.js frontend, talks only to the API.

## 5. Data flow

### Training flow (run on Colab; repeated when the model is improved)

```
Synthetic data generator (Ollama)  ─┐
Hand-curated gold eval set         ─┤→  dataset → push to HF Hub
                                     │
HF Hub dataset → Colab notebook → fine-tune DistilBERT + LoRA
   → evaluate against gold set (MLflow logs metrics)
   → push model + model card to HF Hub
```

### Runtime request flow (every `/analyze` call)

```
User uploads resume + pastes JD  (Next.js)
        │  POST /analyze
        ▼
FastAPI orchestrator:
  1. Ingestion      → ResumeDoc, JobDoc
  2. Extraction     → ResumeProfile, JobProfile        (Ollama)
  3. Scoring        → fit score 0–100                  (fine-tuned HF model)
  4. Similarity     → matched / missing skills map     (sentence-transformers)
  5. Reasoning      → top-3 reasons + 3 bullet rewrites (Ollama)
        │
        ▼
  JSON response → Next.js renders score dial, reasons, rewrite cards
```

The pipeline is **synchronous and linear** — no queues, no async workers. Correct for a
portfolio demo; no infrastructure the project does not need.

### Error handling (only at real boundaries)

- Ingestion: unreadable/empty file → 400 with a clear message.
- Ollama unreachable → 503; the score (step 3) can still return, so the API returns a
  **partial result** rather than failing the whole request.
- HF model load fails on startup → app fails fast (do not serve a broken pipeline).
- Each stage has a timeout; the orchestrator degrades gracefully (score without rewrites)
  rather than hanging.

## 6. Dataset strategy

There is no good public dataset of `(resume, job description)` pairs with fit labels — this is
the highest-risk part of the project, and it is addressed deliberately.

**Training data (synthetic, ~500–1000 pairs):**

- Define ~15–20 job roles across domains (backend dev, data analyst, PM, designer, …).
- For each role, use Ollama to generate JDs at varying seniority.
- Generate resumes deliberately as **strong / partial / weak** fits for chosen JDs — because
  the fit level is requested in the prompt, the label comes for free.
- Vary prompts, temperature, and roles to avoid the classifier learning the generator's tics.
- Push to HF Hub as a versioned dataset.

**Gold eval set (~40–60 pairs, hand-curated):**

- Carefully written or anonymized realistic pairs, manually labeled.
- **Never trained on** — the only metric source that is trusted.

**Honesty as a feature:** the model card documents synthetic provenance openly. Synthetic data
generation is itself an in-demand skill — framed as a deliberate technique, not a shortcut.

**Risk + mitigation:** synthetic data can be repetitive and the classifier may learn generator
artifacts. Mitigation: diversity in generation + always validate against the human gold set.
If the synthetic-vs-gold gap is large, iterate on generation prompts before scaling up.

## 7. Phased build plan

Each phase ships something usable. Sequencing gives an early working artifact and introduces
ML concepts progressively.

| Phase | Build | Deliverable / skill shown |
|---|---|---|
| **0 — Scaffold** | `uv` workspace, Makefile, Next.js + FastAPI hello-world, Ollama installed | `make dev` runs both; monorepo setup |
| **1 — Ingestion** | `pipeline/ingestion`: PDF/DOCX/text → normalized docs + tests | parse a real resume + JD |
| **2 — Data** | Ollama extraction module; synthetic generator; hand-curated gold set; push to HF Hub | versioned dataset on HF profile |
| **3 — Fine-tune** | `packages/training`: DistilBERT + LoRA on Colab, MLflow tracking, eval vs gold set | trained model + model card on HF Hub |
| **4 — Score** | `pipeline/scoring` (loads HF model) + `pipeline/similarity` (sentence-transformers) | pair → score + skill-match map |
| **5 — Reasoning** | `pipeline/reasoning`: Ollama prompts → 3 reasons + 3 rewrites | full pipeline callable as a library |
| **6 — API** | FastAPI `/analyze` wiring all stages, error handling, timeouts, tests | documented working API |
| **7 — Frontend** | Next.js: upload/paste, score dial, reasons, rewrite cards | full-stack app via `make dev` |
| **8 — Deploy** | Docker → HF Spaces (api), Vercel (web), GitHub Actions CI, README + demo GIF | live public demo + writeup |

## 8. Testing

- **Stages 1–5** are pure libraries — unit tested in isolation with sample inputs (sample
  resumes/JDs in `data/`). The scoring stage is also evaluated against the gold set.
- **API** — integration tests against `/analyze` with the pipeline wired up; error-path tests
  for the boundary cases in §5.
- **Model** — evaluation harness in `packages/training` reports metrics against the gold eval
  set; tracked in MLflow across runs.
- **Frontend** — manual verification in a browser for the golden path and edge cases (empty
  upload, Ollama down → partial result).
- **CI** — GitHub Actions runs lint + unit/integration tests on every push.

## 9. Stretch goals (out of core scope)

Only if they earn their place after the core pipeline ships:

- **RAG** over a corpus of real job postings to ground the bullet-rewrite suggestions.
- **Agentic bullet-rewriter** — an agent that iteratively critiques and rewrites bullets.

## 10. Skills demonstrated (portfolio value)

Hugging Face ecosystem (Transformers, Datasets, Hub, PEFT/LoRA, sentence-transformers),
model fine-tuning, synthetic data generation, experiment tracking (MLflow), LLM integration
and prompt engineering, pipeline architecture, FastAPI backend, Next.js full-stack, Docker,
CI/CD, and a deployed live demo — all fully open-source.
