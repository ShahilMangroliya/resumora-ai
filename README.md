# ResumeFit

An open-source AI pipeline that scores a resume against a job description,
explains the fit, and suggests bullet-point rewrites.

See the design doc: `docs/superpowers/specs/2026-05-14-ai-pipeline-design.md`

## Setup

```bash
make install
```

Also see `docs/ollama-setup.md` to install the local LLM runtime.

## Run (dev)

```bash
make dev
```

Boots the FastAPI backend (http://localhost:8000) and the Next.js
frontend (http://localhost:3000) together.

## Test

```bash
make test
```

## Phases

- **Phase 2 — data layer:** see [docs/phase-2-data.md](docs/phase-2-data.md) for the extraction module, synthetic pair generator, gold seed, and the HF Hub publish step.
