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

- **Phase 0 — scaffold:** [plan](docs/superpowers/plans/2026-05-14-phase-0-scaffold.md).
- **Phase 1 — ingestion:** [plan](docs/superpowers/plans/2026-05-14-phase-1-ingestion.md).
- **Phase 2 — data layer:** [plan](docs/superpowers/plans/2026-05-15-phase-2-data.md), [guide](docs/phase-2-data.md).
- **Phase 3 — fine-tune:** [plan](docs/superpowers/plans/2026-05-15-phase-3-finetune.md), [supplement](docs/superpowers/specs/2026-05-15-phase-3-finetune-supplement.md), [guide](docs/phase-3-finetune.md).
