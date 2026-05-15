# Phase 5 — Reasoning: design supplement

**Date:** 2026-05-15
**Status:** Approved
**Supplements:** [`2026-05-14-ai-pipeline-design.md`](2026-05-14-ai-pipeline-design.md) — does not replace it.

This document locks in the Phase 5 decisions that the master spec left open: reasoning API surface, prompt strategy, output shape, error semantics, and how the reasoning module reuses the existing Ollama client. Wherever this supplement and the master spec disagree, the supplement wins for Phase 5 only.

## 1. Scope

Phase 5 ships **one pure library** inside `packages/pipeline`:

- `pipeline.reasoning` — given the Phase 4 score, the Phase 4 skill-match report, the Phase 2 profiles, and the resume's raw text, asks the local Ollama LLM for the **top-3 reasons** for the score and **3 bullet-point rewrites**, returns a validated `ReasoningResult`.

No FastAPI, no orchestration knowledge — `pipeline.reasoning` is imported by the Phase 6 API.

### 1.1 "Full pipeline callable as a library" — clarified

The master spec §7 deliverable for Phase 5 reads "full pipeline callable as a library." This means **every stage (1–5) is importable as a pure Python library**, not that Phase 5 ships a top-level `run_pipeline()` orchestrator. Orchestration lives in `apps/api` (Phase 6). Phase 5's only entry point is `generate_reasoning(...)`.

A Phase 5 smoke test in `packages/pipeline/tests/test_smoke.py` asserts that the five sub-packages — `ingestion`, `extraction`, `scoring`, `similarity`, `reasoning` — are all importable, locking in the "callable as a library" property.

## 2. `pipeline.reasoning`

### 2.1 Public surface

```python
from pipeline.reasoning import (
    BulletRewrite,
    Reason,
    ReasoningError,
    ReasoningResult,
    generate_reasoning,
)

result: ReasoningResult = generate_reasoning(
    score_result=score_result,       # pipeline.scoring.ScoreResult
    skill_report=skill_report,       # pipeline.similarity.SkillMatchReport
    resume_profile=resume_profile,   # pipeline.extraction.ResumeProfile
    job_profile=job_profile,         # pipeline.extraction.JobProfile
    resume_text=resume_doc.raw_text, # str — needed for bullet rewrites
    client=None,                     # injected for tests; default OllamaClient()
)

result.reasons   # list[Reason]         — exactly 3
result.rewrites  # list[BulletRewrite]  — exactly 3
```

The function is **module-level**, not a class, because there is no per-call state to amortize — every invocation runs one Ollama request and returns. This mirrors `extract_resume_profile` / `extract_job_profile` from Phase 2.

### 2.2 Output shape

```python
class Reason(BaseModel):
    """One of the top-3 reasons for the score."""

    summary: str        # short, concrete: "Strong Python skills match required stack"
    evidence: str       # short citation from resume or JD
    category: Literal[
        "matched_skill",
        "missing_skill",
        "experience_match",
        "experience_gap",
        "other",
    ]


class BulletRewrite(BaseModel):
    """A suggested rewrite for a resume bullet."""

    original: str       # the original bullet from the resume; "" if synthesized from profile
    rewritten: str      # the suggested rewrite, tuned for the JD
    rationale: str      # why this rewrite improves the fit


class ReasoningResult(BaseModel):
    reasons: list[Reason]            # validator: exactly 3
    rewrites: list[BulletRewrite]    # validator: exactly 3
```

`reasons` and `rewrites` are pinned to length 3 with field validators. If the LLM returns the wrong count, Pydantic raises `ValidationError` and the reasoner re-raises as `ReasoningError` — no retry-on-shape (see §2.6).

### 2.3 Resume text is required for bullet rewrites

`ResumeProfile.experiences` only carries titles + years; that is too thin to ground a rewrite. The reasoning prompt needs the actual resume bullets, so `resume_text` is a required argument. The function signature documents this and the prompt threads the resume text into the LLM call.

The JD text is **not** passed — `JobProfile.required_skills + nice_to_have_skills + title + seniority` is enough signal for the reasons, and keeping the prompt tight matters for a 3B-class model.

### 2.4 Typed error: `ReasoningError`

`ReasoningError` mirrors `pipeline.extraction.ExtractionError`. It is raised when:
- the Ollama transport fails,
- Ollama returns a non-2xx response that cannot be parsed,
- the JSON shape does not satisfy `ReasoningResult` validation (e.g., wrong reason count).

The master spec §5 says "Ollama unreachable → 503; the score (step 3) can still return, so the API returns a **partial result** rather than failing the whole request." The Phase 6 API catches `ReasoningError` and returns score-only. The typed exception makes that boundary explicit.

### 2.5 Ollama client reuse

`pipeline.reasoning.generate_reasoning` accepts a `client` parameter typed as a `_Client` protocol:

```python
class _Client(Protocol):
    def generate_json(self, prompt: str) -> dict: ...
```

The default client is built from `pipeline.extraction.client.OllamaClient`. This is a **deliberate cross-module reuse**:

- `OllamaClient` is a generic HTTP client — its only extraction-specific configuration is the prompt, which is passed in.
- Moving the client to a shared location (e.g. `pipeline._llm`) would touch Phase 2 code and add scope without changing behavior. Not worth it for a portfolio project.
- The reasoner imports it as an implementation detail; callers that want a different transport pass their own client.

The reuse is documented in the docstring of `generate_reasoning`. If a later phase needs to split the client out, that refactor is local — both call sites use the same Protocol.

### 2.6 No reasoning-level retry

`OllamaClient.generate_json` already retries once on unparseable JSON (Phase 2). The reasoner does **not** add a second retry on `ValidationError` — the failure modes there (wrong reason count, missing field, bad category) are not transient. The Phase 6 API absorbs the typed error and degrades to score-only.

### 2.7 Model default and overrides

The default Ollama model is `llama3.2:3b` (inherited from `OllamaClient`'s default). The prompt is tuned for 3B-class instruct models. Callers can pass a different model by constructing their own `OllamaClient(model="qwen2.5:7b")` and injecting it:

```python
client = OllamaClient(model="qwen2.5:7b")
generate_reasoning(..., client=client)
```

This keeps the model choice in one place (`OllamaClient.__init__`) and avoids leaking the model name through the reasoning surface.

## 3. Prompt strategy

### 3.1 Single-prompt design

One Ollama call produces both reasons and rewrites. Two calls would cost ~2× latency for a slot the user is waiting on, and the two outputs depend on the same context (score, skills, profiles). A single prompt with a strict JSON schema is the right trade.

### 3.2 Prompt template

`pipeline/reasoning/prompts.py` exposes `build_reasoning_prompt(...)` that renders a template with:
- the score, predicted_label, and confidence;
- the matched-required and missing-required skill lists (from `SkillMatchReport`);
- the matched and missing nice-to-have lists;
- the job title, required skills, nice-to-have skills, seniority, and min years;
- the resume titles, skill list, and `total_years_experience`;
- the **raw resume text** (truncated to ~6000 chars to fit a 3B context window).

The instruction block asks for exactly 3 reasons and exactly 3 rewrites, lower-cased category strings from the fixed enum, and JSON-only output (no prose, no markdown). Field names match the Pydantic model.

### 3.3 Resume text truncation

The prompt truncates `resume_text` to 6000 characters before insertion. Llama 3.2 3B has an 8K context window; the rest of the prompt (instructions + structured fields) is well under 2K tokens, so 6000 chars of resume text comfortably fits with margin for the response.

The truncation is naive (head-only) — sufficient for typical 1–2 page resumes (~3–5K chars). A Phase 6+ stretch could swap in smarter truncation that prioritizes recent roles.

### 3.4 Category enum is fixed

The five `Reason.category` values (`matched_skill`, `missing_skill`, `experience_match`, `experience_gap`, `other`) are listed in the prompt and validated by Pydantic. `other` is the escape hatch for the LLM when none of the four specific categories apply — it is rare but worth supporting (avoids forced miscategorization).

## 4. Module layout

```
packages/pipeline/src/pipeline/
└── reasoning/                          ← NEW
    ├── __init__.py                     public surface: generate_reasoning, models, error
    ├── models.py                       Reason, BulletRewrite, ReasoningResult
    ├── errors.py                       ReasoningError
    ├── prompts.py                      build_reasoning_prompt
    └── reasoner.py                     generate_reasoning() module function
```

Tests mirror this layout under `packages/pipeline/tests/reasoning/`:

```
packages/pipeline/tests/reasoning/
├── __init__.py                         empty marker
├── test_models.py                      strict-3 validation, category enum
├── test_prompts.py                     prompt threads fields, truncates resume
├── test_reasoner.py                    fake client tests for happy + error paths
└── test_integration.py                 @pytest.mark.integration — real Ollama call
```

## 5. Dependencies added

None. Phase 5 reuses `httpx` (already a runtime dep from Phase 2), `pydantic` (Phase 0+), and the Phase 2 `OllamaClient`. No new entries in `packages/pipeline/pyproject.toml`.

## 6. Testing strategy

- **Unit tests inject a fake client** — same pattern as Phase 2 extraction. Tests assert that the prompt threads expected fields and that the returned shape is validated. Wrong-count and out-of-enum payloads raise `ReasoningError`.
- **One integration test, gated** with `@pytest.mark.integration` — calls a real local Ollama instance, runs the full prompt, and asserts the response is shape-valid. Skipped by default; consistent with Phase 2/3/4.
- **Smoke test** in `packages/pipeline/tests/test_smoke.py` asserts that every pipeline sub-package (`ingestion`, `extraction`, `scoring`, `similarity`, `reasoning`) is importable.

## 7. Out of Phase 5 scope

- **No API wiring** — Phase 6.
- **No streaming** — `OllamaClient` uses `stream=False`; reasoning returns the full payload in one shot. Streaming is a Phase 7 frontend stretch.
- **No multi-turn or critique loop** — the agentic bullet rewriter is listed as a stretch goal in the master spec §9; Phase 5 ships single-shot generation.
- **No automated quality eval** of reasons/rewrites — there is no good public benchmark, and human review is the right loop for this stage. A `data/reasoning-eval/` folder of curated examples is a Phase 6+ task.
- **No prompt versioning system** — the prompt is a Python string. If/when it becomes versioned, a separate `prompts.yaml` or `prompts/v2.py` is the natural shape.

## 8. Prerequisites tracked in the plan

The Phase 5 plan opens with two prerequisites:

1. **Ollama installed and `llama3.2:3b` pulled** — same prerequisite as Phase 2 extraction. The integration test requires a live Ollama at `http://localhost:11434`. See `docs/ollama-setup.md`.
2. **Phases 2 + 4 models available** — `ResumeProfile`, `JobProfile`, `ScoreResult`, `SkillMatchReport` are all in place (already shipped). Unit tests construct these directly; no Ollama, no Phase 3 model needed.
