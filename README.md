# ResumeFit

An open-source AI pipeline that scores a resume against a job description,
explains the fit, and suggests bullet-point rewrites.

See the design doc: `docs/superpowers/specs/2026-05-14-ai-pipeline-design.md`

## Setup

```bash
make install
```

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
