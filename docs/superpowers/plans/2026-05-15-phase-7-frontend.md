# Phase 7 — Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Next.js single-page UI at `/` that uploads a resume + pastes a JD, posts to `POST /analyze`, and renders the score, reasons, bullet rewrites, skill-match map, and partial-result warnings.

**Architecture:** A client component on the home page owns form state and result state. A tiny `lib/` module mirrors the API schemas in TypeScript and wraps `fetch`. Result rendering is split into focused components: `<ScoreDial>`, `<ReasonsList>`, `<RewriteCards>`, `<SkillMatchPanel>`, `<WarningsBanner>`. Loading / empty / error states render in place of the result. No state library — `useState` is sufficient at this scale.

**Tech Stack:** Next.js 16 (App Router, already scaffolded), React 19, TypeScript (strict), Tailwind v4 (already installed). `fetch` for HTTP — no client library. No JS test framework: the master spec §8 specifies manual browser verification for the frontend. Correctness signals are `tsc --noEmit`, ESLint, and the live UI.

**This plan is Phase 7 only.** It follows the master design doc `docs/superpowers/specs/2026-05-14-ai-pipeline-design.md` (§4 component "apps/web", §5 runtime flow, §7 phase 7, §8 testing). Decisions locked in during planning on 2026-05-15:

- **Client component (`"use client"`) on the home page.** The form, the fetch, and the result all live on the client. SSR adds no value for a single-user-driven request and would complicate streaming the file upload.
- **No JS test framework in Phase 7.** The master spec §8 commits to manual browser verification for the frontend. Adding Vitest + RTL is non-trivial setup with limited payoff at this scale. TypeScript strict + ESLint cover the static-correctness floor; the dev server with the API behind it covers the rest. If we later need component tests, that's a follow-on.
- **API base URL from `NEXT_PUBLIC_API_URL`, default `http://localhost:8000`.** Matches the CORS default in `apps/api/src/api/config.py` (`http://localhost:3000`). The API already serves CORS for the dev origin.
- **Score display: raw number + label badge, not a rescaled gauge.** The classifier emits scores in [20, 85] (see `ScoreResult.score` bounds in `packages/pipeline/src/pipeline/scoring/models.py`). Re-mapping to 0–100 would lie about the model's actual output. Show the raw number, color it by `predicted_label` (weak red / partial amber / strong green), and place a horizontal progress bar from 20 → 85.
- **Conditional rendering for partial results.** `skill_report` and `reasoning` are nullable on the wire. When they are `null` the panel does not render at all — the `<WarningsBanner>` carries the explanation. The score always renders because the API always returns it.
- **One client-only home page, no routes.** No `/results`, no shareable links, no history. The spec is "upload + see result"; anything else is scope creep.
- **No file-size client-side validation.** The API enforces what it accepts. Mirroring caps in the UI doubles maintenance for a portfolio demo. The form does require the file and a non-empty JD before posting — that is UX, not security.
- **Tailwind for styling, no design system library.** Tailwind v4 is already configured. Hand-roll components for visual control; portfolio polish comes from the layout, type, and color, not from a component kit.

> **Prerequisite:** Phase 6 API is running. From the repo root, `make dev` starts both servers; alternatively `uv run --package api uvicorn api.main:app --reload --port 8000` in one shell and `npm --prefix apps/web run dev` in another.

---

## File Structure

Files created or modified in this phase:

- `apps/web/.env.example` — **create**: documents `NEXT_PUBLIC_API_URL`.
- `apps/web/src/lib/env.ts` — **create**: read `NEXT_PUBLIC_API_URL` once with a default.
- `apps/web/src/lib/types.ts` — **create**: TypeScript mirrors of `AnalyzeResponse` and nested models.
- `apps/web/src/lib/api.ts` — **create**: `analyzeResume({resume, jobDescription})` and `ApiError`.
- `apps/web/src/app/layout.tsx` — **modify**: replace boilerplate metadata with Resumora AI title/description.
- `apps/web/src/app/page.tsx` — **rewrite**: thin server-component shell that renders the client `<Analyzer>`.
- `apps/web/src/components/Analyzer.tsx` — **create**: top-level client component owning all state and the fetch call.
- `apps/web/src/components/AnalyzeForm.tsx` — **create**: file input + JD textarea + submit; calls a parent-supplied `onSubmit`.
- `apps/web/src/components/ScoreDial.tsx` — **create**: large score number + label badge + 20→85 progress bar.
- `apps/web/src/components/ReasonsList.tsx` — **create**: ordered list of top-3 reasons with category badges.
- `apps/web/src/components/RewriteCards.tsx` — **create**: three side-by-side cards: original → rewritten → rationale.
- `apps/web/src/components/SkillMatchPanel.tsx` — **create**: required matched/missing + nice-to-have matched/missing as four chip groups; match-rate readout.
- `apps/web/src/components/WarningsBanner.tsx` — **create**: amber banner listing any `warnings[]`.
- `apps/web/src/components/EmptyState.tsx` — **create**: pre-submit hint copy.
- `apps/web/src/app/globals.css` — **modify**: extend the existing `@theme` block with brand colors for weak/partial/strong + readable body font stack.
- `apps/web/README.md` — **rewrite**: replace `create-next-app` boilerplate with Resumora AI-specific instructions.
- `docs/phase-7-frontend.md` — **create**: user-facing Phase 7 guide.
- `README.md` — **modify**: add Phase 7 entry.

---

## Task 1: Frontend env config + README cleanup

**Files:**
- Create: `apps/web/.env.example`
- Rewrite: `apps/web/README.md`
- Modify: `apps/web/src/app/layout.tsx`
- Modify: `.gitignore` (only if it does not already ignore `apps/web/.env.local`)

- [ ] **Step 1: Create `apps/web/.env.example`**

```
# Base URL of the Resumora AI FastAPI backend.
# Local dev default matches `make dev` (Procfile binds the API to port 8000).
NEXT_PUBLIC_API_URL=http://localhost:8000
```

- [ ] **Step 2: Verify `apps/web/.env.local` is git-ignored**

Run: `git check-ignore apps/web/.env.local || true`

Expected: prints the path (ignored) — Next.js's standard `.gitignore` already covers `.env*.local`. If it does not, append `apps/web/.env*.local` to the repo-root `.gitignore`. Do not commit any `.env.local` files.

- [ ] **Step 3: Rewrite `apps/web/README.md`**

Replace the entire file with:

````markdown
# Resumora AI — Web

Next.js 16 frontend for Resumora AI. Talks to the FastAPI backend in
`apps/api` via `POST /analyze`.

## Env

Copy `.env.example` → `.env.local` and adjust if your API is not on
`http://localhost:8000`:

```bash
cp .env.example .env.local
```

| Var | Default | Notes |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Base URL of the Resumora AI API. |

## Run

From the repo root, the recommended path is:

```bash
make dev
```

This boots the API on :8000 and the web app on :3000 together via
`honcho` (see `Procfile`).

Or directly from this directory:

```bash
npm install
npm run dev
```

Open <http://localhost:3000>.

## Build / lint

```bash
npm run build
npm run lint
```
````

- [ ] **Step 4: Update layout metadata**

In `apps/web/src/app/layout.tsx`, replace:

```tsx
export const metadata: Metadata = {
  title: "Create Next App",
  description: "Generated by create next app",
};
```

with:

```tsx
export const metadata: Metadata = {
  title: "Resumora AI",
  description: "Score a resume against a job description and improve it.",
};
```

- [ ] **Step 5: Commit**

```bash
git add apps/web/.env.example apps/web/README.md apps/web/src/app/layout.tsx
git commit -m "chore(web): wire env, metadata, and README for Phase 7"
```

---

## Task 2: TypeScript mirrors of API schemas

**Files:**
- Create: `apps/web/src/lib/types.ts`

These types are the single source of truth on the frontend for what `POST /analyze` returns. They mirror the Pydantic models in `apps/api/src/api/schemas.py` and the pipeline package.

- [ ] **Step 1: Write `apps/web/src/lib/types.ts`**

```ts
export type PredictedLabel = "weak" | "partial" | "strong";

export interface ScoreResult {
  score: number;
  confidence: number;
  class_probabilities: Record<PredictedLabel, number>;
  predicted_label: PredictedLabel;
}

export interface SkillMatch {
  jd_skill: string;
  resume_skill: string;
  similarity: number;
  matched: boolean;
}

export interface SkillMatchReport {
  required_matched: SkillMatch[];
  required_missing: SkillMatch[];
  nice_to_have_matched: SkillMatch[];
  nice_to_have_missing: SkillMatch[];
  match_rate: number;
}

export type ReasonCategory =
  | "matched_skill"
  | "missing_skill"
  | "experience_match"
  | "experience_gap"
  | "other";

export interface Reason {
  summary: string;
  evidence: string;
  category: ReasonCategory;
}

export interface BulletRewrite {
  original: string;
  rewritten: string;
  rationale: string;
}

export interface ReasoningResult {
  reasons: Reason[];
  rewrites: BulletRewrite[];
}

export interface AnalyzeResponse {
  score: ScoreResult;
  skill_report: SkillMatchReport | null;
  reasoning: ReasoningResult | null;
  warnings: string[];
}
```

- [ ] **Step 2: Type-check**

Run from `apps/web`: `npx tsc --noEmit`

Expected: PASS (no errors).

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/lib/types.ts
git commit -m "feat(web): add TypeScript mirrors of AnalyzeResponse schemas"
```

---

## Task 3: API client

**Files:**
- Create: `apps/web/src/lib/env.ts`
- Create: `apps/web/src/lib/api.ts`

The client builds a `FormData`, posts it, and either returns `AnalyzeResponse` or throws an `ApiError` carrying the HTTP status and a user-readable message.

- [ ] **Step 1: Write `apps/web/src/lib/env.ts`**

```ts
const DEFAULT_API_URL = "http://localhost:8000";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? DEFAULT_API_URL).replace(/\/$/, "");
```

- [ ] **Step 2: Write `apps/web/src/lib/api.ts`**

```ts
import { API_URL } from "./env";
import type { AnalyzeResponse } from "./types";

export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

export interface AnalyzeInput {
  resume: File;
  jobDescription: string;
}

export async function analyzeResume({ resume, jobDescription }: AnalyzeInput): Promise<AnalyzeResponse> {
  const body = new FormData();
  body.append("resume", resume);
  body.append("job_description", jobDescription);

  let response: Response;
  try {
    response = await fetch(`${API_URL}/analyze`, { method: "POST", body });
  } catch (cause) {
    throw new ApiError(0, "Could not reach the Resumora AI API. Is the backend running?");
  }

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}.`;
    try {
      const payload = (await response.json()) as { detail?: unknown };
      if (typeof payload.detail === "string") detail = payload.detail;
    } catch {
      // body wasn't JSON; keep the generic message
    }
    throw new ApiError(response.status, detail);
  }

  return (await response.json()) as AnalyzeResponse;
}
```

- [ ] **Step 3: Type-check**

Run from `apps/web`: `npx tsc --noEmit`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add apps/web/src/lib/env.ts apps/web/src/lib/api.ts
git commit -m "feat(web): add analyzeResume API client with typed errors"
```

---

## Task 4: Global styles and theme tokens

**Files:**
- Modify: `apps/web/src/app/globals.css`

Add brand-colored tokens for the three score labels and a readable body font stack. Keep the existing `@theme inline` block; just extend it.

- [ ] **Step 1: Rewrite `apps/web/src/app/globals.css`**

Replace the whole file with:

```css
@import "tailwindcss";

:root {
  --background: #fafaf9;
  --foreground: #1c1917;
  --surface: #ffffff;
  --border: #e7e5e4;
  --muted: #78716c;

  --score-weak: #dc2626;
  --score-partial: #d97706;
  --score-strong: #16a34a;
}

@theme inline {
  --color-background: var(--background);
  --color-foreground: var(--foreground);
  --color-surface: var(--surface);
  --color-border: var(--border);
  --color-muted: var(--muted);
  --color-score-weak: var(--score-weak);
  --color-score-partial: var(--score-partial);
  --color-score-strong: var(--score-strong);
  --font-sans: var(--font-geist-sans);
  --font-mono: var(--font-geist-mono);
}

@media (prefers-color-scheme: dark) {
  :root {
    --background: #0c0a09;
    --foreground: #f5f5f4;
    --surface: #1c1917;
    --border: #292524;
    --muted: #a8a29e;
  }
}

body {
  background: var(--background);
  color: var(--foreground);
  font-family: var(--font-sans), ui-sans-serif, system-ui, sans-serif;
}
```

- [ ] **Step 2: Commit**

```bash
git add apps/web/src/app/globals.css
git commit -m "feat(web): extend theme with score-label colors and surface tokens"
```

---

## Task 5: `AnalyzeForm` component

**Files:**
- Create: `apps/web/src/components/AnalyzeForm.tsx`

A controlled form. Parent component owns the result and loading state; this component only emits `(resume, jobDescription)` via `onSubmit` and renders a `disabled` state when `pending` is true.

- [ ] **Step 1: Write `apps/web/src/components/AnalyzeForm.tsx`**

```tsx
"use client";

import { useState } from "react";

interface AnalyzeFormProps {
  pending: boolean;
  onSubmit: (input: { resume: File; jobDescription: string }) => void;
}

export function AnalyzeForm({ pending, onSubmit }: AnalyzeFormProps) {
  const [resume, setResume] = useState<File | null>(null);
  const [jobDescription, setJobDescription] = useState("");

  const canSubmit = resume !== null && jobDescription.trim().length > 0 && !pending;

  function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!resume || !jobDescription.trim()) return;
    onSubmit({ resume, jobDescription });
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-6">
      <label className="flex flex-col gap-2">
        <span className="text-sm font-medium">Resume</span>
        <input
          type="file"
          accept=".pdf,.docx,.txt"
          onChange={(event) => setResume(event.target.files?.[0] ?? null)}
          className="block w-full rounded-md border border-[color:var(--border)] bg-[color:var(--surface)] px-3 py-2 text-sm file:mr-3 file:rounded file:border-0 file:bg-zinc-900 file:px-3 file:py-1.5 file:text-sm file:text-white hover:file:bg-zinc-800 dark:file:bg-zinc-100 dark:file:text-zinc-900"
          required
        />
        <span className="text-xs text-[color:var(--muted)]">PDF, DOCX, or plain text.</span>
      </label>

      <label className="flex flex-col gap-2">
        <span className="text-sm font-medium">Job description</span>
        <textarea
          value={jobDescription}
          onChange={(event) => setJobDescription(event.target.value)}
          rows={10}
          placeholder="Paste the job description here…"
          className="block w-full resize-y rounded-md border border-[color:var(--border)] bg-[color:var(--surface)] px-3 py-2 text-sm leading-6 focus:outline-none focus:ring-2 focus:ring-zinc-500"
          required
        />
      </label>

      <button
        type="submit"
        disabled={!canSubmit}
        className="self-start rounded-md bg-zinc-900 px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-zinc-700 disabled:cursor-not-allowed disabled:bg-zinc-400 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200"
      >
        {pending ? "Analyzing…" : "Analyze"}
      </button>
    </form>
  );
}
```

- [ ] **Step 2: Type-check**

Run from `apps/web`: `npx tsc --noEmit`

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/components/AnalyzeForm.tsx
git commit -m "feat(web): add AnalyzeForm with file upload and JD textarea"
```

---

## Task 6: `ScoreDial` component

**Files:**
- Create: `apps/web/src/components/ScoreDial.tsx`

Visual: large rounded score number, label chip, and a horizontal progress bar showing where the score lands in the model's true [20, 85] output range. Color is driven by `predicted_label`.

- [ ] **Step 1: Write `apps/web/src/components/ScoreDial.tsx`**

```tsx
import type { ScoreResult } from "@/lib/types";

const SCORE_MIN = 20;
const SCORE_MAX = 85;

const LABEL_COLOR: Record<ScoreResult["predicted_label"], string> = {
  weak: "var(--color-score-weak)",
  partial: "var(--color-score-partial)",
  strong: "var(--color-score-strong)",
};

const LABEL_COPY: Record<ScoreResult["predicted_label"], string> = {
  weak: "Weak fit",
  partial: "Partial fit",
  strong: "Strong fit",
};

export function ScoreDial({ score }: { score: ScoreResult }) {
  const color = LABEL_COLOR[score.predicted_label];
  const pct = Math.max(0, Math.min(1, (score.score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN))) * 100;
  const confidencePct = Math.round(score.confidence * 100);

  return (
    <section
      aria-label="Fit score"
      className="flex flex-col gap-4 rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)] p-6"
    >
      <div className="flex items-baseline gap-4">
        <span className="text-6xl font-semibold tabular-nums" style={{ color }}>
          {score.score.toFixed(1)}
        </span>
        <span className="text-sm text-[color:var(--muted)]">/ 100</span>
        <span
          className="ml-auto rounded-full px-3 py-1 text-xs font-medium uppercase tracking-wide text-white"
          style={{ backgroundColor: color }}
        >
          {LABEL_COPY[score.predicted_label]}
        </span>
      </div>

      <div className="flex flex-col gap-2">
        <div className="relative h-2 w-full overflow-hidden rounded-full bg-[color:var(--border)]">
          <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color }} />
        </div>
        <div className="flex justify-between text-xs text-[color:var(--muted)]">
          <span>{SCORE_MIN}</span>
          <span>{SCORE_MAX}</span>
        </div>
      </div>

      <p className="text-xs text-[color:var(--muted)]">
        Confidence {confidencePct}% — weak {(score.class_probabilities.weak * 100).toFixed(0)}%, partial{" "}
        {(score.class_probabilities.partial * 100).toFixed(0)}%, strong{" "}
        {(score.class_probabilities.strong * 100).toFixed(0)}%.
      </p>
    </section>
  );
}
```

- [ ] **Step 2: Type-check**

Run from `apps/web`: `npx tsc --noEmit`

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/components/ScoreDial.tsx
git commit -m "feat(web): add ScoreDial showing raw score and label"
```

---

## Task 7: `ReasonsList` component

**Files:**
- Create: `apps/web/src/components/ReasonsList.tsx`

- [ ] **Step 1: Write `apps/web/src/components/ReasonsList.tsx`**

```tsx
import type { Reason } from "@/lib/types";

const CATEGORY_LABEL: Record<Reason["category"], string> = {
  matched_skill: "Match",
  missing_skill: "Gap",
  experience_match: "Experience",
  experience_gap: "Experience gap",
  other: "Note",
};

const CATEGORY_TONE: Record<Reason["category"], string> = {
  matched_skill: "bg-emerald-100 text-emerald-900 dark:bg-emerald-900/40 dark:text-emerald-200",
  missing_skill: "bg-rose-100 text-rose-900 dark:bg-rose-900/40 dark:text-rose-200",
  experience_match: "bg-emerald-100 text-emerald-900 dark:bg-emerald-900/40 dark:text-emerald-200",
  experience_gap: "bg-amber-100 text-amber-900 dark:bg-amber-900/40 dark:text-amber-200",
  other: "bg-zinc-100 text-zinc-900 dark:bg-zinc-800 dark:text-zinc-200",
};

export function ReasonsList({ reasons }: { reasons: Reason[] }) {
  return (
    <section className="flex flex-col gap-3 rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)] p-6">
      <h2 className="text-lg font-semibold">Why this score</h2>
      <ol className="flex flex-col gap-3">
        {reasons.map((reason, index) => (
          <li
            key={index}
            className="flex flex-col gap-1 rounded-xl border border-[color:var(--border)] p-4"
          >
            <div className="flex items-center justify-between gap-3">
              <p className="text-sm font-medium">{reason.summary}</p>
              <span className={`shrink-0 rounded-full px-2.5 py-0.5 text-xs ${CATEGORY_TONE[reason.category]}`}>
                {CATEGORY_LABEL[reason.category]}
              </span>
            </div>
            <p className="text-sm text-[color:var(--muted)]">{reason.evidence}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}
```

- [ ] **Step 2: Type-check**

Run from `apps/web`: `npx tsc --noEmit`

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/components/ReasonsList.tsx
git commit -m "feat(web): add ReasonsList with category-tagged reasons"
```

---

## Task 8: `RewriteCards` component

**Files:**
- Create: `apps/web/src/components/RewriteCards.tsx`

- [ ] **Step 1: Write `apps/web/src/components/RewriteCards.tsx`**

```tsx
import type { BulletRewrite } from "@/lib/types";

export function RewriteCards({ rewrites }: { rewrites: BulletRewrite[] }) {
  return (
    <section className="flex flex-col gap-3 rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)] p-6">
      <h2 className="text-lg font-semibold">Bullet rewrites</h2>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        {rewrites.map((rewrite, index) => (
          <article
            key={index}
            className="flex flex-col gap-3 rounded-xl border border-[color:var(--border)] p-4"
          >
            {rewrite.original ? (
              <div className="flex flex-col gap-1">
                <span className="text-xs uppercase tracking-wide text-[color:var(--muted)]">Before</span>
                <p className="text-sm line-through decoration-[color:var(--muted)]/70">{rewrite.original}</p>
              </div>
            ) : (
              <span className="text-xs uppercase tracking-wide text-[color:var(--muted)]">Suggested addition</span>
            )}
            <div className="flex flex-col gap-1">
              <span className="text-xs uppercase tracking-wide text-[color:var(--muted)]">After</span>
              <p className="text-sm font-medium">{rewrite.rewritten}</p>
            </div>
            <p className="text-xs text-[color:var(--muted)]">{rewrite.rationale}</p>
          </article>
        ))}
      </div>
    </section>
  );
}
```

- [ ] **Step 2: Type-check**

Run from `apps/web`: `npx tsc --noEmit`

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/components/RewriteCards.tsx
git commit -m "feat(web): add RewriteCards rendering original→rewrite→rationale"
```

---

## Task 9: `SkillMatchPanel` component

**Files:**
- Create: `apps/web/src/components/SkillMatchPanel.tsx`

Four chip groups — required matched, required missing, nice-to-have matched, nice-to-have missing — plus the `match_rate` readout. Empty groups simply do not render.

- [ ] **Step 1: Write `apps/web/src/components/SkillMatchPanel.tsx`**

```tsx
import type { SkillMatch, SkillMatchReport } from "@/lib/types";

interface ChipGroupProps {
  title: string;
  skills: SkillMatch[];
  tone: "match" | "miss";
}

function ChipGroup({ title, skills, tone }: ChipGroupProps) {
  if (skills.length === 0) return null;
  const chipClass =
    tone === "match"
      ? "bg-emerald-100 text-emerald-900 dark:bg-emerald-900/40 dark:text-emerald-200"
      : "bg-rose-100 text-rose-900 dark:bg-rose-900/40 dark:text-rose-200";
  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-xs font-medium uppercase tracking-wide text-[color:var(--muted)]">{title}</h3>
      <ul className="flex flex-wrap gap-1.5">
        {skills.map((skill) => (
          <li
            key={`${skill.jd_skill}|${skill.resume_skill}`}
            title={
              tone === "match"
                ? `Matched to "${skill.resume_skill}" (sim ${skill.similarity.toFixed(2)})`
                : skill.resume_skill
                  ? `Closest in resume: "${skill.resume_skill}" (sim ${skill.similarity.toFixed(2)})`
                  : "Not found in resume"
            }
            className={`rounded-full px-2.5 py-1 text-xs ${chipClass}`}
          >
            {skill.jd_skill}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function SkillMatchPanel({ report }: { report: SkillMatchReport }) {
  const matchPct = Math.round(report.match_rate * 100);
  return (
    <section className="flex flex-col gap-5 rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)] p-6">
      <div className="flex items-baseline justify-between gap-3">
        <h2 className="text-lg font-semibold">Skill match</h2>
        <span className="text-sm text-[color:var(--muted)]">
          {matchPct}% of required skills matched
        </span>
      </div>
      <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
        <ChipGroup title="Required — matched" skills={report.required_matched} tone="match" />
        <ChipGroup title="Required — missing" skills={report.required_missing} tone="miss" />
        <ChipGroup title="Nice to have — matched" skills={report.nice_to_have_matched} tone="match" />
        <ChipGroup title="Nice to have — missing" skills={report.nice_to_have_missing} tone="miss" />
      </div>
    </section>
  );
}
```

- [ ] **Step 2: Type-check**

Run from `apps/web`: `npx tsc --noEmit`

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/components/SkillMatchPanel.tsx
git commit -m "feat(web): add SkillMatchPanel with required/nice-to-have chips"
```

---

## Task 10: `WarningsBanner` + `EmptyState` components

**Files:**
- Create: `apps/web/src/components/WarningsBanner.tsx`
- Create: `apps/web/src/components/EmptyState.tsx`

- [ ] **Step 1: Write `apps/web/src/components/WarningsBanner.tsx`**

```tsx
export function WarningsBanner({ warnings }: { warnings: string[] }) {
  if (warnings.length === 0) return null;
  return (
    <aside
      role="status"
      className="flex flex-col gap-2 rounded-xl border border-amber-300 bg-amber-50 p-4 text-amber-900 dark:border-amber-700/60 dark:bg-amber-900/30 dark:text-amber-100"
    >
      <p className="text-sm font-medium">Partial result</p>
      <ul className="list-disc pl-5 text-sm">
        {warnings.map((message, index) => (
          <li key={index}>{message}</li>
        ))}
      </ul>
    </aside>
  );
}
```

- [ ] **Step 2: Write `apps/web/src/components/EmptyState.tsx`**

```tsx
export function EmptyState() {
  return (
    <section className="flex flex-col items-center gap-3 rounded-2xl border border-dashed border-[color:var(--border)] bg-[color:var(--surface)] p-12 text-center">
      <h2 className="text-base font-semibold">No analysis yet</h2>
      <p className="max-w-md text-sm text-[color:var(--muted)]">
        Upload a resume and paste a job description above, then click Analyze. We&rsquo;ll score
        the fit, explain why, and suggest three bullet rewrites.
      </p>
    </section>
  );
}
```

- [ ] **Step 3: Type-check**

Run from `apps/web`: `npx tsc --noEmit`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add apps/web/src/components/WarningsBanner.tsx apps/web/src/components/EmptyState.tsx
git commit -m "feat(web): add WarningsBanner and EmptyState"
```

---

## Task 11: `Analyzer` client container

**Files:**
- Create: `apps/web/src/components/Analyzer.tsx`

The single client component that owns state and stitches everything together.

- [ ] **Step 1: Write `apps/web/src/components/Analyzer.tsx`**

```tsx
"use client";

import { useState } from "react";

import { analyzeResume, ApiError } from "@/lib/api";
import type { AnalyzeResponse } from "@/lib/types";

import { AnalyzeForm } from "./AnalyzeForm";
import { EmptyState } from "./EmptyState";
import { ReasonsList } from "./ReasonsList";
import { RewriteCards } from "./RewriteCards";
import { ScoreDial } from "./ScoreDial";
import { SkillMatchPanel } from "./SkillMatchPanel";
import { WarningsBanner } from "./WarningsBanner";

type Status =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "success"; result: AnalyzeResponse };

export function Analyzer() {
  const [status, setStatus] = useState<Status>({ kind: "idle" });

  async function handleSubmit(input: { resume: File; jobDescription: string }) {
    setStatus({ kind: "loading" });
    try {
      const result = await analyzeResume(input);
      setStatus({ kind: "success", result });
    } catch (cause) {
      const message =
        cause instanceof ApiError
          ? cause.message
          : "Something went wrong while analyzing. Please try again.";
      setStatus({ kind: "error", message });
    }
  }

  return (
    <div className="flex flex-col gap-10">
      <section className="rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)] p-6">
        <AnalyzeForm pending={status.kind === "loading"} onSubmit={handleSubmit} />
      </section>

      {status.kind === "idle" && <EmptyState />}

      {status.kind === "loading" && (
        <section
          aria-live="polite"
          className="flex items-center justify-center rounded-2xl border border-dashed border-[color:var(--border)] bg-[color:var(--surface)] p-12 text-sm text-[color:var(--muted)]"
        >
          Scoring resume — this may take up to 90 seconds while the model runs.
        </section>
      )}

      {status.kind === "error" && (
        <aside
          role="alert"
          className="rounded-xl border border-rose-300 bg-rose-50 p-4 text-sm text-rose-900 dark:border-rose-700/60 dark:bg-rose-900/30 dark:text-rose-100"
        >
          {status.message}
        </aside>
      )}

      {status.kind === "success" && (
        <div className="flex flex-col gap-6">
          <WarningsBanner warnings={status.result.warnings} />
          <ScoreDial score={status.result.score} />
          {status.result.reasoning && <ReasonsList reasons={status.result.reasoning.reasons} />}
          {status.result.skill_report && <SkillMatchPanel report={status.result.skill_report} />}
          {status.result.reasoning && <RewriteCards rewrites={status.result.reasoning.rewrites} />}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Type-check**

Run from `apps/web`: `npx tsc --noEmit`

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/components/Analyzer.tsx
git commit -m "feat(web): add Analyzer client container with idle/loading/error/success"
```

---

## Task 12: Rewrite the home page shell

**Files:**
- Rewrite: `apps/web/src/app/page.tsx`

Server component shell: hero title, subtitle, and the `<Analyzer>` client component.

- [ ] **Step 1: Rewrite `apps/web/src/app/page.tsx`**

Replace the whole file with:

```tsx
import { Analyzer } from "@/components/Analyzer";

export default function Home() {
  return (
    <main className="mx-auto flex w-full max-w-4xl flex-col gap-10 px-6 py-16">
      <header className="flex flex-col gap-3">
        <span className="text-xs font-medium uppercase tracking-[0.2em] text-[color:var(--muted)]">
          Resumora AI
        </span>
        <h1 className="text-3xl font-semibold tracking-tight md:text-4xl">
          Score your resume against any job description.
        </h1>
        <p className="max-w-2xl text-sm text-[color:var(--muted)] md:text-base">
          Upload a resume, paste a job description, and get a fit score, the top three
          reasons behind it, and three bullet rewrites you can lift into your CV.
        </p>
      </header>

      <Analyzer />
    </main>
  );
}
```

- [ ] **Step 2: Verify production build still passes**

Run from `apps/web`: `npx tsc --noEmit`

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/app/page.tsx
git commit -m "feat(web): replace home page with Resumora AI shell and Analyzer"
```

---

## Task 13: Lint, build, and manual browser verification

**Files:**
- (no code changes — verification only)

Per the design doc §8, the frontend's correctness signal is manual browser verification on the golden path and the partial-result edge cases.

- [ ] **Step 1: Lint**

Run from `apps/web`: `npm run lint`

Expected: PASS (no errors). Address any errors before continuing — warnings about unused imports or stale boilerplate are fine to fix in this task.

- [ ] **Step 2: Production build**

Run from `apps/web`: `npm run build`

Expected: a successful build, no TypeScript errors, no static-analysis errors.

- [ ] **Step 3: Boot the full stack**

From the repo root, in one shell: `make dev`

Expected: API on <http://localhost:8000/health> returns `{"status":"ok"}`; the web app at <http://localhost:3000> renders the Resumora AI shell with the form.

- [ ] **Step 4: Golden-path verification (Ollama up, scorer up)**

In the browser at <http://localhost:3000>:
1. Pick any resume file from `data/` (e.g. one of the synthetic resumes or a real PDF you have locally).
2. Paste a JD into the textarea.
3. Click Analyze.
4. Verify:
   - The button shows "Analyzing…" and is disabled.
   - On success, `<ScoreDial>` shows a 20–85 score, label badge, progress bar, and confidence breakdown.
   - `<ReasonsList>` shows three reasons with category badges.
   - `<SkillMatchPanel>` shows four chip groups (or fewer if some are empty) and a `match_rate` percentage.
   - `<RewriteCards>` shows three rewrite cards.
   - No `<WarningsBanner>`.

- [ ] **Step 5: Partial-result verification (Ollama down)**

Stop Ollama (`pkill ollama` on macOS, or simply leave it off if it was not running). Re-submit the form. Verify:
- The button completes (not stuck).
- `<ScoreDial>` still renders.
- `<WarningsBanner>` appears with the "Profile extraction failed; downstream stages skipped" message.
- `<ReasonsList>`, `<RewriteCards>`, `<SkillMatchPanel>` do **not** render.

Restart Ollama afterwards: `ollama serve` (or whatever the user runs locally).

- [ ] **Step 6: Validation edge cases**

- Empty JD: Submit button is disabled.
- No file: Submit button is disabled.
- Submit a `.zip` or other unsupported extension: API returns 400; the page shows the red error banner with the parser's message.
- API not running: kill the API process, submit, verify the banner reads "Could not reach the Resumora AI API. Is the backend running?" Restart the API.

- [ ] **Step 7: Commit any cleanup that surfaced during verification**

If you fixed any lint/build issues in steps 1–6, commit them now. If nothing changed, skip this step.

```bash
git add -A
git commit -m "chore(web): post-verification cleanup" # only if there are changes
```

---

## Task 14: Documentation

**Files:**
- Create: `docs/phase-7-frontend.md`
- Modify: `README.md`

- [ ] **Step 1: Write `docs/phase-7-frontend.md`**

```markdown
# Phase 7 — Frontend

A Next.js 16 single-page UI that drives the Resumora AI pipeline through
the FastAPI backend.

## What it does

- Uploads a resume (PDF / DOCX / plain text) and a pasted JD.
- POSTs to `POST /analyze` as `multipart/form-data`.
- Renders the result: fit score with a label band, top-3 reasons, skill-match
  chips, three bullet rewrites, and any partial-result warnings.

## Configuration

| Var | Default | Notes |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Base URL of the Resumora AI API. |

Configured per-environment via `apps/web/.env.local` (copied from
`apps/web/.env.example`). For a deployment, set this on the hosting provider
(Vercel project env, HF Space env, etc.).

CORS on the API side is controlled by `RESUMORA_AI_CORS_ORIGINS`
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

There is no JS test framework in Phase 7 — per the master spec §8 the frontend
relies on:

- `npx tsc --noEmit` — strict type-check.
- `npm run lint` — ESLint.
- `npm run build` — production build.
- Manual browser verification on the golden path and partial-result edge cases
  (see the Phase 7 plan, Task 13).
```

- [ ] **Step 2: Add the Phase 7 entry to the root `README.md`**

In `README.md`, in the Phases list, append after the Phase 6 entry:

```markdown
- **Phase 7 — Frontend:** [plan](docs/superpowers/plans/2026-05-15-phase-7-frontend.md), [guide](docs/phase-7-frontend.md).
```

- [ ] **Step 3: Commit**

```bash
git add docs/phase-7-frontend.md README.md
git commit -m "docs: add Phase 7 frontend user guide and README entry"
```

---

## Self-Review

**Spec coverage:**
- §4 component "apps/web — Next.js frontend, talks only to the API" → Tasks 2–12 build it; Task 13 verifies it.
- §5 runtime request flow ("Next.js renders score dial, reasons, rewrite cards") → Tasks 6–8.
- §5 error handling ("partial result rather than failing the whole request") → Task 10 (`WarningsBanner`) + Task 11 (`Analyzer` conditional rendering).
- §7 phase 7 deliverable ("Next.js: upload/paste, score dial, reasons, rewrite cards; full-stack app via `make dev`") → Task 13 step 3.
- §8 testing for frontend ("manual verification in a browser for the golden path and edge cases (empty upload, Ollama down → partial result)") → Task 13 steps 4–6.

**Placeholder scan:** No `TBD`, `TODO`, `implement later`, or generic "add error handling" steps. Each step has the exact code or command needed.

**Type consistency:** `AnalyzeResponse`, `ScoreResult`, `SkillMatchReport`, `ReasoningResult`, `Reason`, `BulletRewrite`, `SkillMatch` defined in Task 2 are referenced verbatim by Tasks 3 and 5–11. The `analyzeResume({ resume, jobDescription })` signature defined in Task 3 matches the `onSubmit` payload in Task 5 (`AnalyzeForm`) and the call site in Task 11 (`Analyzer`).
