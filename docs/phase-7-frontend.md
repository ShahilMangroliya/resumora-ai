# Phase 7 — Frontend

A Next.js 16 single-page UI that drives the ResumeFit pipeline through
the FastAPI backend.

## What it does

- Uploads a resume (PDF / DOCX / plain text) and a pasted JD.
- POSTs to `POST /analyze` as `multipart/form-data`.
- Renders the result: fit score with a label band, top-3 reasons, skill-match
  chips, three bullet rewrites, and any partial-result warnings.

## Configuration

| Var | Default | Notes |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Base URL of the ResumeFit API. |

Configured per-environment via `apps/web/.env.local` (copied from
`apps/web/.env.example`). For a deployment, set this on the hosting provider
(Vercel project env, HF Space env, etc.).

CORS on the API side is controlled by `RESUMEFIT_CORS_ORIGINS`
(see `docs/phase-6-api.md`).

## Run locally

From the repo root:

```bash
make dev
```

Boots the FastAPI API on :8000 and the Next.js app on :3000.

## States

| State | Trigger | UI |
|---|---|---|
| Idle | Initial load | Form + empty-state hint card |
| Loading | Submit clicked | Form disabled, "Scoring resume…" panel |
| Error | Network failure or non-2xx response | Red banner with the API message |
| Full result | 200 + `warnings == []` | Score, reasons, skill match, rewrites |
| Partial result | 200 + `warnings` non-empty | Amber banner; only sections with data render |

## Files

- `src/app/page.tsx` — server component shell (title + `<Analyzer>`).
- `src/components/Analyzer.tsx` — client container, owns form + result state.
- `src/components/AnalyzeForm.tsx` — file + textarea + submit.
- `src/components/ScoreDial.tsx`, `ReasonsList.tsx`, `RewriteCards.tsx`,
  `SkillMatchPanel.tsx`, `WarningsBanner.tsx`, `EmptyState.tsx` — result panels.
- `src/lib/api.ts` — `analyzeResume(...)` and `ApiError`.
- `src/lib/types.ts` — TS mirrors of the API schemas.
- `src/lib/env.ts` — reads `NEXT_PUBLIC_API_URL`.

## Verification

- `npm test` — Vitest + jsdom + React Testing Library. Covers the API client
  (success, parsed `detail` on 4xx, generic message on non-JSON error body,
  network-failure ApiError) and the result panels' branch logic (empty-state,
  partial-result, warnings banner, conditional sections). Also covers the
  `Analyzer` submit flow with a mocked `analyzeResume`.
- `npx tsc --noEmit` — strict type-check.
- `npm run lint` — ESLint.
- `npm run build` — production build.
- Manual browser verification on the golden path and partial-result edge cases
  (see the Phase 7 plan, Task 13).

### Why `fireEvent.submit` and not `userEvent.click` in tests

`userEvent.click` on a `type="submit"` button does not reliably fire the form's
`submit` event under React 19 + jsdom. The `Analyzer` test submits via
`fireEvent.submit(form)` instead. This is documented inline in
`Analyzer.test.tsx`.
