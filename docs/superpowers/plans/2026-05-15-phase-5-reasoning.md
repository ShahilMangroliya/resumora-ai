# Phase 5 — Reasoning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `packages/pipeline/src/pipeline/reasoning/` — a single pure library that turns `(ScoreResult, SkillMatchReport, ResumeProfile, JobProfile, resume_text)` into a `ReasoningResult` of 3 reasons + 3 bullet rewrites via one Ollama call.

**Architecture:** Module-level function `generate_reasoning(...)` (no class — no per-call state worth amortizing), accepting an injected `_Client` Protocol whose default is `pipeline.extraction.client.OllamaClient`. Returns a Pydantic `ReasoningResult` with strict-3 validators on both lists. Errors collapse into a typed `ReasoningError` so the Phase 6 API can `except` it and degrade to score-only.

**Tech Stack:** Python 3.12, `pydantic`, `httpx` (via `OllamaClient`). Tests use `pytest` with the existing `not integration` default marker pattern; injected fake client mirrors the Phase 2 extraction tests.

**This plan is Phase 5 only.** It follows the master design doc `docs/superpowers/specs/2026-05-14-ai-pipeline-design.md` (§4 component 5, §7 phase 5) and its supplement `docs/superpowers/specs/2026-05-15-phase-5-reasoning-supplement.md`. Decisions locked in during brainstorming on 2026-05-15:

- **Single library, single phase** — `pipeline.reasoning` with module-level `generate_reasoning`.
- **Resume text is a required input** — `ResumeProfile.experiences` is too thin to rewrite bullets from; the function takes `resume_text` directly.
- **Strict 3 + 3 output via Pydantic validators** — invalid shapes raise `ReasoningError`. No reasoning-level retry; `OllamaClient.generate_json` already retries once on unparseable JSON.
- **Single Ollama call** for both reasons and rewrites — keeps latency low and shared context coherent.
- **Reuse `OllamaClient` from `pipeline.extraction.client`** — deliberate cross-module reuse; documented in the supplement §2.5. No refactor of Phase 2 code.
- **Typed `ReasoningError`** — mirrors `ExtractionError`; the Phase 6 API uses it to return partial results when Ollama is unreachable.
- **`Reason.category` is a fixed enum** of `{matched_skill, missing_skill, experience_match, experience_gap, other}`.
- **Resume text truncated to 6000 chars** in the prompt to fit a 3B-class model's context window.
- **Integration test gated** with `@pytest.mark.integration` — consistent with Phase 2/3/4.

> **Prerequisite:** Ollama installed locally with `llama3.2:3b` pulled (see `docs/ollama-setup.md`). The unit tests do not require Ollama — they inject a fake client. Only the gated integration test (Task 5 step 7) hits real Ollama.

---

## File Structure

Files created or modified in this phase:

- `packages/pipeline/src/pipeline/reasoning/__init__.py` — **create**: public surface.
- `packages/pipeline/src/pipeline/reasoning/models.py` — **create**: `Reason`, `BulletRewrite`, `ReasoningResult` + `REASON_CATEGORIES`.
- `packages/pipeline/src/pipeline/reasoning/errors.py` — **create**: `ReasoningError`.
- `packages/pipeline/src/pipeline/reasoning/prompts.py` — **create**: `build_reasoning_prompt` + `MAX_RESUME_CHARS`.
- `packages/pipeline/src/pipeline/reasoning/reasoner.py` — **create**: `generate_reasoning(...)`.
- `packages/pipeline/tests/reasoning/__init__.py` — **create**: empty marker.
- `packages/pipeline/tests/reasoning/test_models.py` — **create**: strict-3 validators + category enum.
- `packages/pipeline/tests/reasoning/test_prompts.py` — **create**: prompt threads expected fields; truncates resume.
- `packages/pipeline/tests/reasoning/test_reasoner.py` — **create**: fake-client happy + error paths.
- `packages/pipeline/tests/reasoning/test_integration.py` — **create**: gated live-Ollama test.
- `packages/pipeline/tests/test_smoke.py` — **modify**: add `pipeline.reasoning` to the import smoke check.
- `docs/phase-5-reasoning.md` — **create**: user-facing Phase 5 guide.
- `README.md` — **modify**: add Phase 5 entry to the phases list.

---

## Task 1: Reasoning sub-package skeleton

**Files:**
- Create: `packages/pipeline/src/pipeline/reasoning/__init__.py`
- Create: `packages/pipeline/tests/reasoning/__init__.py`

- [ ] **Step 1: Create the reasoning sub-package marker**

Create `packages/pipeline/src/pipeline/reasoning/__init__.py` with this exact content (re-exports added in Task 5):

```python
"""Phase 5 reasoning library: turn score + skill report + profiles into reasons and bullet rewrites."""
```

- [ ] **Step 2: Create the reasoning tests marker**

Create `packages/pipeline/tests/reasoning/__init__.py` as a zero-byte file (the marker pytest needs to discover the sub-package).

- [ ] **Step 3: Sanity-check that pytest still collects the existing suites**

Run: `uv run pytest --collect-only -q`
Expected: collection succeeds with no errors; the new empty directories add no test cases yet.

- [ ] **Step 4: Commit**

```bash
git add packages/pipeline/src/pipeline/reasoning/__init__.py packages/pipeline/tests/reasoning/__init__.py
git commit -m "feat: add Phase 5 reasoning sub-package skeleton"
```

---

## Task 2: Reasoning errors module

**Files:**
- Create: `packages/pipeline/src/pipeline/reasoning/errors.py`

`ReasoningError` is a tiny module so the import graph stays acyclic — `reasoner.py` imports it; `models.py` does not.

- [ ] **Step 1: Implement `errors.py`**

Create `packages/pipeline/src/pipeline/reasoning/errors.py`:

```python
class ReasoningError(Exception):
    """Raised when the reasoning stage cannot produce a result.

    Covers Ollama transport failures, timeouts, unparseable model output,
    and JSON payloads that do not satisfy the `ReasoningResult` shape.
    The API layer maps this to a partial-result response (score-only).
    """
```

- [ ] **Step 2: Verify the import**

Run:
```bash
uv run python -c "from pipeline.reasoning.errors import ReasoningError; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add packages/pipeline/src/pipeline/reasoning/errors.py
git commit -m "feat: add ReasoningError typed exception"
```

---

## Task 3: Reasoning Pydantic models

**Files:**
- Create: `packages/pipeline/src/pipeline/reasoning/models.py`
- Create: `packages/pipeline/tests/reasoning/test_models.py`

`Reason`, `BulletRewrite`, `ReasoningResult` are the validated shapes the reasoner returns. `ReasoningResult` enforces exactly 3 reasons and exactly 3 rewrites via field validators.

- [ ] **Step 1: Write the failing tests**

Create `packages/pipeline/tests/reasoning/test_models.py`:

```python
import pytest
from pydantic import ValidationError

from pipeline.reasoning.models import (
    REASON_CATEGORIES,
    BulletRewrite,
    Reason,
    ReasoningResult,
)


def test_reason_categories_constant():
    assert REASON_CATEGORIES == (
        "matched_skill",
        "missing_skill",
        "experience_match",
        "experience_gap",
        "other",
    )


def test_reason_valid():
    r = Reason(
        summary="Strong Python skills match required stack",
        evidence="Resume lists 5 years of Python; JD requires senior Python.",
        category="matched_skill",
    )
    assert r.category == "matched_skill"


def test_reason_invalid_category_rejected():
    with pytest.raises(ValidationError):
        Reason(summary="x", evidence="y", category="excellent")


def test_bullet_rewrite_valid():
    b = BulletRewrite(
        original="- Built things with Python",
        rewritten="- Built and shipped 3 production services in Python (FastAPI, ~10k req/s).",
        rationale="Quantifies impact and names the framework the JD requires.",
    )
    assert b.rewritten.startswith("-")


def test_bullet_rewrite_empty_original_allowed_for_synthesis():
    # When the LLM synthesizes a new bullet rather than rewriting one, original=""
    b = BulletRewrite(original="", rewritten="- New synthesized bullet.", rationale="why")
    assert b.original == ""


def _three_reasons() -> list[Reason]:
    return [
        Reason(summary=f"Reason {i}", evidence=f"Evidence {i}", category="matched_skill")
        for i in range(3)
    ]


def _three_rewrites() -> list[BulletRewrite]:
    return [
        BulletRewrite(original=f"orig {i}", rewritten=f"rewritten {i}", rationale=f"r {i}")
        for i in range(3)
    ]


def test_reasoning_result_valid_with_strict_three():
    r = ReasoningResult(reasons=_three_reasons(), rewrites=_three_rewrites())
    assert len(r.reasons) == 3
    assert len(r.rewrites) == 3


def test_reasoning_result_rejects_fewer_than_three_reasons():
    with pytest.raises(ValidationError):
        ReasoningResult(reasons=_three_reasons()[:2], rewrites=_three_rewrites())


def test_reasoning_result_rejects_more_than_three_reasons():
    with pytest.raises(ValidationError):
        ReasoningResult(reasons=_three_reasons() + [_three_reasons()[0]], rewrites=_three_rewrites())


def test_reasoning_result_rejects_fewer_than_three_rewrites():
    with pytest.raises(ValidationError):
        ReasoningResult(reasons=_three_reasons(), rewrites=_three_rewrites()[:2])


def test_reasoning_result_rejects_more_than_three_rewrites():
    with pytest.raises(ValidationError):
        ReasoningResult(reasons=_three_reasons(), rewrites=_three_rewrites() + [_three_rewrites()[0]])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest packages/pipeline/tests/reasoning/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.reasoning.models'`.

- [ ] **Step 3: Implement `models.py`**

Create `packages/pipeline/src/pipeline/reasoning/models.py`:

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ReasonCategory = Literal[
    "matched_skill",
    "missing_skill",
    "experience_match",
    "experience_gap",
    "other",
]

REASON_CATEGORIES: tuple[ReasonCategory, ...] = (
    "matched_skill",
    "missing_skill",
    "experience_match",
    "experience_gap",
    "other",
)


class Reason(BaseModel):
    """One of the top-3 reasons explaining the fit score."""

    summary: str = Field(description="Short, concrete one-line reason.")
    evidence: str = Field(description="Short citation from the resume or JD.")
    category: ReasonCategory


class BulletRewrite(BaseModel):
    """A suggested rewrite of a resume bullet for a specific JD."""

    original: str = Field(
        description="The original bullet from the resume. Empty string when synthesized.",
    )
    rewritten: str = Field(description="The suggested rewrite, tuned for the JD.")
    rationale: str = Field(description="Why this rewrite improves the fit.")


class ReasoningResult(BaseModel):
    """Output of `generate_reasoning(...)` — exactly 3 reasons and 3 rewrites."""

    reasons: list[Reason] = Field(
        min_length=3, max_length=3, description="Exactly three reasons.",
    )
    rewrites: list[BulletRewrite] = Field(
        min_length=3, max_length=3, description="Exactly three bullet rewrites.",
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest packages/pipeline/tests/reasoning/test_models.py -v`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/pipeline/src/pipeline/reasoning/models.py packages/pipeline/tests/reasoning/test_models.py
git commit -m "feat: add reasoning pydantic models with strict-3 validators"
```

---

## Task 4: Reasoning prompt builder

**Files:**
- Create: `packages/pipeline/src/pipeline/reasoning/prompts.py`
- Create: `packages/pipeline/tests/reasoning/test_prompts.py`

`build_reasoning_prompt(...)` renders the single prompt that asks the LLM for both reasons and rewrites in one shot. The resume text is truncated to `MAX_RESUME_CHARS` (6000) to fit a 3B-class context window.

- [ ] **Step 1: Write the failing tests**

Create `packages/pipeline/tests/reasoning/test_prompts.py`:

```python
from pipeline.extraction.models import Experience, JobProfile, ResumeProfile
from pipeline.reasoning.prompts import MAX_RESUME_CHARS, build_reasoning_prompt
from pipeline.scoring.models import ScoreResult
from pipeline.similarity.models import SkillMatch, SkillMatchReport


def _score_result() -> ScoreResult:
    return ScoreResult(
        score=72.5,
        confidence=0.84,
        class_probabilities={"weak": 0.05, "partial": 0.11, "strong": 0.84},
        predicted_label="strong",
    )


def _skill_report() -> SkillMatchReport:
    return SkillMatchReport(
        required_matched=[
            SkillMatch(jd_skill="python", resume_skill="Python", similarity=0.99, matched=True),
        ],
        required_missing=[
            SkillMatch(jd_skill="kubernetes", resume_skill="docker", similarity=0.42, matched=False),
        ],
        nice_to_have_matched=[
            SkillMatch(jd_skill="docker", resume_skill="docker", similarity=0.99, matched=True),
        ],
        nice_to_have_missing=[
            SkillMatch(jd_skill="terraform", resume_skill="aws", similarity=0.35, matched=False),
        ],
        match_rate=0.5,
    )


def _resume_profile() -> ResumeProfile:
    return ResumeProfile(
        titles=["Backend Engineer"],
        skills=["python", "fastapi", "docker"],
        experiences=[Experience(title="Backend Engineer", years=3.0)],
        education=["BSc Computer Science"],
        total_years_experience=3.0,
    )


def _job_profile() -> JobProfile:
    return JobProfile(
        title="Senior Backend Engineer",
        required_skills=["python", "kubernetes"],
        nice_to_have_skills=["docker", "terraform"],
        seniority="senior",
        min_years_experience=5.0,
    )


def test_prompt_threads_score_label_and_confidence():
    prompt = build_reasoning_prompt(
        score_result=_score_result(),
        skill_report=_skill_report(),
        resume_profile=_resume_profile(),
        job_profile=_job_profile(),
        resume_text="some resume text",
    )
    assert "72.5" in prompt
    assert "strong" in prompt
    # Confidence shown as either 0.84 or 84%
    assert "0.84" in prompt or "84" in prompt


def test_prompt_includes_matched_and_missing_skills():
    prompt = build_reasoning_prompt(
        score_result=_score_result(),
        skill_report=_skill_report(),
        resume_profile=_resume_profile(),
        job_profile=_job_profile(),
        resume_text="some resume text",
    )
    assert "python" in prompt
    assert "kubernetes" in prompt
    assert "terraform" in prompt
    assert "docker" in prompt


def test_prompt_includes_job_title_and_seniority():
    prompt = build_reasoning_prompt(
        score_result=_score_result(),
        skill_report=_skill_report(),
        resume_profile=_resume_profile(),
        job_profile=_job_profile(),
        resume_text="some resume text",
    )
    assert "Senior Backend Engineer" in prompt
    assert "senior" in prompt
    assert "5" in prompt  # min_years_experience


def test_prompt_includes_resume_text_when_short():
    prompt = build_reasoning_prompt(
        score_result=_score_result(),
        skill_report=_skill_report(),
        resume_profile=_resume_profile(),
        job_profile=_job_profile(),
        resume_text="UNIQUE-MARKER-XYZ-123",
    )
    assert "UNIQUE-MARKER-XYZ-123" in prompt


def test_prompt_truncates_long_resume_text():
    long_text = "a" * (MAX_RESUME_CHARS + 5000)
    prompt = build_reasoning_prompt(
        score_result=_score_result(),
        skill_report=_skill_report(),
        resume_profile=_resume_profile(),
        job_profile=_job_profile(),
        resume_text=long_text,
    )
    # The 'a' run in the prompt should be at most MAX_RESUME_CHARS long.
    run = max((len(s) for s in prompt.split() if set(s) == {"a"}), default=0)
    assert run <= MAX_RESUME_CHARS


def test_prompt_asks_for_strict_three_each():
    prompt = build_reasoning_prompt(
        score_result=_score_result(),
        skill_report=_skill_report(),
        resume_profile=_resume_profile(),
        job_profile=_job_profile(),
        resume_text="x",
    )
    # The prompt explicitly asks for three.
    assert "3" in prompt or "three" in prompt.lower()
    # And mentions both keys.
    assert "reasons" in prompt
    assert "rewrites" in prompt


def test_prompt_lists_allowed_categories():
    prompt = build_reasoning_prompt(
        score_result=_score_result(),
        skill_report=_skill_report(),
        resume_profile=_resume_profile(),
        job_profile=_job_profile(),
        resume_text="x",
    )
    for cat in ("matched_skill", "missing_skill", "experience_match", "experience_gap", "other"):
        assert cat in prompt


def test_prompt_handles_empty_skill_lists():
    empty_report = SkillMatchReport(
        required_matched=[],
        required_missing=[],
        nice_to_have_matched=[],
        nice_to_have_missing=[],
        match_rate=1.0,
    )
    empty_job = JobProfile(
        title="x",
        required_skills=[],
        nice_to_have_skills=[],
        seniority="mid",
        min_years_experience=0.0,
    )
    # Should not raise.
    prompt = build_reasoning_prompt(
        score_result=_score_result(),
        skill_report=empty_report,
        resume_profile=_resume_profile(),
        job_profile=empty_job,
        resume_text="x",
    )
    assert isinstance(prompt, str)
    assert len(prompt) > 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest packages/pipeline/tests/reasoning/test_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.reasoning.prompts'`.

- [ ] **Step 3: Implement `prompts.py`**

Create `packages/pipeline/src/pipeline/reasoning/prompts.py`:

```python
from __future__ import annotations

from pipeline.extraction.models import JobProfile, ResumeProfile
from pipeline.scoring.models import ScoreResult
from pipeline.similarity.models import SkillMatch, SkillMatchReport

MAX_RESUME_CHARS = 6000


_TEMPLATE = """You are an expert technical recruiter explaining why a resume fits a job and suggesting concrete improvements.

You will receive (1) a fit score with confidence, (2) a skill match report, (3) the candidate's profile, (4) the job's profile, and (5) the candidate's resume text.

Return ONLY a JSON object with this exact shape — no prose, no markdown:

{{
  "reasons": [
    {{"summary": "string", "evidence": "string", "category": "matched_skill|missing_skill|experience_match|experience_gap|other"}},
    {{"summary": "string", "evidence": "string", "category": "matched_skill|missing_skill|experience_match|experience_gap|other"}},
    {{"summary": "string", "evidence": "string", "category": "matched_skill|missing_skill|experience_match|experience_gap|other"}}
  ],
  "rewrites": [
    {{"original": "string", "rewritten": "string", "rationale": "string"}},
    {{"original": "string", "rewritten": "string", "rationale": "string"}},
    {{"original": "string", "rewritten": "string", "rationale": "string"}}
  ]
}}

Rules:
- Return exactly 3 reasons and exactly 3 rewrites — no more, no fewer.
- "category" must be one of: matched_skill, missing_skill, experience_match, experience_gap, other.
- Reasons should be specific (cite concrete skills, years, or roles) — never generic platitudes.
- Each rewrite picks an existing resume bullet ("original") and improves it for this specific JD ("rewritten"), then explains why ("rationale"). If no bullet fits, set "original" to "" and synthesize a new bullet.
- Rewritten bullets should lead with an action verb, include a metric where plausible, and reference a skill or technology the JD requires.

Fit score: {score:.1f}/100  (predicted: {label}, confidence: {confidence:.2f})
Required match rate: {match_rate:.0%}

Matched required skills: {req_matched}
Missing required skills: {req_missing}
Matched nice-to-have skills: {nice_matched}
Missing nice-to-have skills: {nice_missing}

Job:
- title: {job_title}
- seniority: {seniority}
- min years experience: {min_years}
- required skills: {required_skills}
- nice-to-have skills: {nice_skills}

Candidate profile:
- titles: {resume_titles}
- skills: {resume_skills}
- total years experience: {total_years}
- education: {education}

Resume text:
---
{resume_text}
---
"""


def _skill_list(matches: list[SkillMatch]) -> str:
    if not matches:
        return "(none)"
    return ", ".join(m.jd_skill for m in matches)


def build_reasoning_prompt(
    *,
    score_result: ScoreResult,
    skill_report: SkillMatchReport,
    resume_profile: ResumeProfile,
    job_profile: JobProfile,
    resume_text: str,
) -> str:
    """Render the single prompt that asks the LLM for 3 reasons + 3 rewrites."""
    truncated_resume = resume_text[:MAX_RESUME_CHARS]
    return _TEMPLATE.format(
        score=score_result.score,
        label=score_result.predicted_label,
        confidence=score_result.confidence,
        match_rate=skill_report.match_rate,
        req_matched=_skill_list(skill_report.required_matched),
        req_missing=_skill_list(skill_report.required_missing),
        nice_matched=_skill_list(skill_report.nice_to_have_matched),
        nice_missing=_skill_list(skill_report.nice_to_have_missing),
        job_title=job_profile.title,
        seniority=job_profile.seniority,
        min_years=job_profile.min_years_experience,
        required_skills=", ".join(job_profile.required_skills) or "(none)",
        nice_skills=", ".join(job_profile.nice_to_have_skills) or "(none)",
        resume_titles=", ".join(resume_profile.titles) or "(none)",
        resume_skills=", ".join(resume_profile.skills) or "(none)",
        total_years=resume_profile.total_years_experience,
        education=", ".join(resume_profile.education) or "(none)",
        resume_text=truncated_resume,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest packages/pipeline/tests/reasoning/test_prompts.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/pipeline/src/pipeline/reasoning/prompts.py packages/pipeline/tests/reasoning/test_prompts.py
git commit -m "feat: add reasoning prompt builder with resume truncation"
```

---

## Task 5: `generate_reasoning` entry point + public surface

**Files:**
- Create: `packages/pipeline/src/pipeline/reasoning/reasoner.py`
- Modify: `packages/pipeline/src/pipeline/reasoning/__init__.py`
- Create: `packages/pipeline/tests/reasoning/test_reasoner.py`
- Create: `packages/pipeline/tests/reasoning/test_integration.py`

`generate_reasoning(...)` calls the injected client with the rendered prompt and validates the JSON payload into a `ReasoningResult`. All failure modes — transport, parse, validation — collapse into `ReasoningError`.

- [ ] **Step 1: Write the failing unit tests**

Create `packages/pipeline/tests/reasoning/test_reasoner.py`:

```python
from dataclasses import dataclass, field

import pytest

from pipeline.extraction.errors import ExtractionError
from pipeline.extraction.models import Experience, JobProfile, ResumeProfile
from pipeline.reasoning import (
    BulletRewrite,
    Reason,
    ReasoningError,
    ReasoningResult,
    generate_reasoning,
)
from pipeline.scoring.models import ScoreResult
from pipeline.similarity.models import SkillMatch, SkillMatchReport


@dataclass
class FakeClient:
    payloads: list[dict | Exception]
    seen_prompts: list[str] = field(default_factory=list)

    def generate_json(self, prompt: str) -> dict:
        self.seen_prompts.append(prompt)
        head = self.payloads.pop(0)
        if isinstance(head, Exception):
            raise head
        return head


def _score() -> ScoreResult:
    return ScoreResult(
        score=72.0,
        confidence=0.8,
        class_probabilities={"weak": 0.1, "partial": 0.1, "strong": 0.8},
        predicted_label="strong",
    )


def _report() -> SkillMatchReport:
    return SkillMatchReport(
        required_matched=[
            SkillMatch(jd_skill="python", resume_skill="Python", similarity=0.99, matched=True)
        ],
        required_missing=[],
        nice_to_have_matched=[],
        nice_to_have_missing=[],
        match_rate=1.0,
    )


def _resume_profile() -> ResumeProfile:
    return ResumeProfile(
        titles=["Backend Engineer"],
        skills=["python"],
        experiences=[Experience(title="Backend Engineer", years=3.0)],
        education=["BSc"],
        total_years_experience=3.0,
    )


def _job_profile() -> JobProfile:
    return JobProfile(
        title="Senior Backend Engineer",
        required_skills=["python"],
        nice_to_have_skills=[],
        seniority="senior",
        min_years_experience=5.0,
    )


def _valid_payload() -> dict:
    return {
        "reasons": [
            {"summary": "s1", "evidence": "e1", "category": "matched_skill"},
            {"summary": "s2", "evidence": "e2", "category": "missing_skill"},
            {"summary": "s3", "evidence": "e3", "category": "experience_gap"},
        ],
        "rewrites": [
            {"original": "o1", "rewritten": "r1", "rationale": "why1"},
            {"original": "o2", "rewritten": "r2", "rationale": "why2"},
            {"original": "", "rewritten": "r3", "rationale": "why3"},
        ],
    }


def test_generate_reasoning_returns_validated_result():
    client = FakeClient(payloads=[_valid_payload()])
    result = generate_reasoning(
        score_result=_score(),
        skill_report=_report(),
        resume_profile=_resume_profile(),
        job_profile=_job_profile(),
        resume_text="some resume text",
        client=client,
    )
    assert isinstance(result, ReasoningResult)
    assert len(result.reasons) == 3
    assert len(result.rewrites) == 3
    assert isinstance(result.reasons[0], Reason)
    assert isinstance(result.rewrites[2], BulletRewrite)
    assert result.reasons[1].category == "missing_skill"


def test_generate_reasoning_threads_resume_text_into_prompt():
    client = FakeClient(payloads=[_valid_payload()])
    generate_reasoning(
        score_result=_score(),
        skill_report=_report(),
        resume_profile=_resume_profile(),
        job_profile=_job_profile(),
        resume_text="UNIQUE-RESUME-MARKER-9182",
        client=client,
    )
    assert "UNIQUE-RESUME-MARKER-9182" in client.seen_prompts[0]


def test_generate_reasoning_raises_on_wrong_reason_count():
    bad = _valid_payload()
    bad["reasons"] = bad["reasons"][:2]
    client = FakeClient(payloads=[bad])
    with pytest.raises(ReasoningError, match="invalid"):
        generate_reasoning(
            score_result=_score(),
            skill_report=_report(),
            resume_profile=_resume_profile(),
            job_profile=_job_profile(),
            resume_text="x",
            client=client,
        )


def test_generate_reasoning_raises_on_wrong_rewrite_count():
    bad = _valid_payload()
    bad["rewrites"] = bad["rewrites"] + [{"original": "", "rewritten": "x", "rationale": "y"}]
    client = FakeClient(payloads=[bad])
    with pytest.raises(ReasoningError, match="invalid"):
        generate_reasoning(
            score_result=_score(),
            skill_report=_report(),
            resume_profile=_resume_profile(),
            job_profile=_job_profile(),
            resume_text="x",
            client=client,
        )


def test_generate_reasoning_raises_on_bad_category():
    bad = _valid_payload()
    bad["reasons"][0]["category"] = "excellent"
    client = FakeClient(payloads=[bad])
    with pytest.raises(ReasoningError, match="invalid"):
        generate_reasoning(
            score_result=_score(),
            skill_report=_report(),
            resume_profile=_resume_profile(),
            job_profile=_job_profile(),
            resume_text="x",
            client=client,
        )


def test_generate_reasoning_wraps_extraction_error_as_reasoning_error():
    # If the underlying OllamaClient raises ExtractionError, the reasoner re-raises
    # as ReasoningError so callers only need to catch one type.
    client = FakeClient(payloads=[ExtractionError("Ollama unreachable: boom")])
    with pytest.raises(ReasoningError, match="Ollama"):
        generate_reasoning(
            score_result=_score(),
            skill_report=_report(),
            resume_profile=_resume_profile(),
            job_profile=_job_profile(),
            resume_text="x",
            client=client,
        )
```

- [ ] **Step 2: Write the failing integration test**

Create `packages/pipeline/tests/reasoning/test_integration.py`:

```python
import pytest

from pipeline.extraction.models import Experience, JobProfile, ResumeProfile
from pipeline.reasoning import ReasoningResult, generate_reasoning
from pipeline.scoring.models import ScoreResult
from pipeline.similarity.models import SkillMatch, SkillMatchReport


@pytest.mark.integration
def test_generate_reasoning_against_live_ollama():
    """Calls a local Ollama. Skipped unless `pytest -m integration` is used."""
    score = ScoreResult(
        score=68.0,
        confidence=0.7,
        class_probabilities={"weak": 0.1, "partial": 0.2, "strong": 0.7},
        predicted_label="strong",
    )
    report = SkillMatchReport(
        required_matched=[
            SkillMatch(jd_skill="python", resume_skill="Python", similarity=0.98, matched=True),
        ],
        required_missing=[
            SkillMatch(jd_skill="kubernetes", resume_skill="docker", similarity=0.4, matched=False),
        ],
        nice_to_have_matched=[],
        nice_to_have_missing=[],
        match_rate=0.5,
    )
    resume_profile = ResumeProfile(
        titles=["Backend Engineer"],
        skills=["python", "fastapi", "docker"],
        experiences=[Experience(title="Backend Engineer", years=3.0)],
        education=["BSc Computer Science"],
        total_years_experience=3.0,
    )
    job_profile = JobProfile(
        title="Senior Backend Engineer",
        required_skills=["python", "kubernetes"],
        nice_to_have_skills=["aws"],
        seniority="senior",
        min_years_experience=5.0,
    )
    resume_text = (
        "Backend Engineer with 3 years of Python and FastAPI experience.\n"
        "- Built and shipped a real-time analytics service in Python\n"
        "- Containerized 8 services with Docker and deployed to AWS\n"
        "- Wrote integration tests with pytest at 90% coverage\n"
    )
    result = generate_reasoning(
        score_result=score,
        skill_report=report,
        resume_profile=resume_profile,
        job_profile=job_profile,
        resume_text=resume_text,
    )
    assert isinstance(result, ReasoningResult)
    assert len(result.reasons) == 3
    assert len(result.rewrites) == 3
    for r in result.reasons:
        assert r.summary
        assert r.category in (
            "matched_skill",
            "missing_skill",
            "experience_match",
            "experience_gap",
            "other",
        )
    for b in result.rewrites:
        assert b.rewritten
```

- [ ] **Step 3: Run the unit tests to verify they fail**

Run: `uv run pytest packages/pipeline/tests/reasoning/test_reasoner.py -v`
Expected: FAIL with `ImportError` from `pipeline.reasoning`.

- [ ] **Step 4: Implement `reasoner.py`**

Create `packages/pipeline/src/pipeline/reasoning/reasoner.py`:

```python
from __future__ import annotations

from typing import Protocol

from pydantic import ValidationError

from pipeline.extraction.client import OllamaClient
from pipeline.extraction.errors import ExtractionError
from pipeline.extraction.models import JobProfile, ResumeProfile
from pipeline.reasoning.errors import ReasoningError
from pipeline.reasoning.models import ReasoningResult
from pipeline.reasoning.prompts import build_reasoning_prompt
from pipeline.scoring.models import ScoreResult
from pipeline.similarity.models import SkillMatchReport


class _Client(Protocol):
    def generate_json(self, prompt: str) -> dict: ...


def _default_client() -> _Client:
    return OllamaClient()


def generate_reasoning(
    *,
    score_result: ScoreResult,
    skill_report: SkillMatchReport,
    resume_profile: ResumeProfile,
    job_profile: JobProfile,
    resume_text: str,
    client: _Client | None = None,
) -> ReasoningResult:
    """Generate 3 reasons + 3 bullet rewrites for a scored (resume, JD) pair.

    Uses a local Ollama LLM via the injected `client`. The default client is
    `pipeline.extraction.client.OllamaClient` — deliberate cross-module reuse
    (see Phase 5 supplement §2.5). Pass a custom client to swap models.

    Raises `ReasoningError` on transport failure, unparseable JSON, or any
    payload that does not satisfy the `ReasoningResult` shape. The Phase 6
    API catches `ReasoningError` and degrades to a score-only response.
    """
    client = client or _default_client()
    prompt = build_reasoning_prompt(
        score_result=score_result,
        skill_report=skill_report,
        resume_profile=resume_profile,
        job_profile=job_profile,
        resume_text=resume_text,
    )
    try:
        payload = client.generate_json(prompt)
    except ExtractionError as exc:
        raise ReasoningError(str(exc)) from exc

    try:
        return ReasoningResult.model_validate(payload)
    except ValidationError as exc:
        raise ReasoningError(f"Ollama returned invalid ReasoningResult: {exc}") from exc
```

- [ ] **Step 5: Replace the `__init__.py` to expose the public surface**

Replace `packages/pipeline/src/pipeline/reasoning/__init__.py` with:

```python
"""Phase 5 reasoning library: turn score + skill report + profiles into reasons and bullet rewrites."""

from pipeline.reasoning.errors import ReasoningError
from pipeline.reasoning.models import (
    REASON_CATEGORIES,
    BulletRewrite,
    Reason,
    ReasonCategory,
    ReasoningResult,
)
from pipeline.reasoning.reasoner import generate_reasoning

__all__ = [
    "REASON_CATEGORIES",
    "BulletRewrite",
    "Reason",
    "ReasonCategory",
    "ReasoningError",
    "ReasoningResult",
    "generate_reasoning",
]
```

- [ ] **Step 6: Run the unit tests to verify they pass**

Run: `uv run pytest packages/pipeline/tests/reasoning/test_reasoner.py -v`
Expected: PASS (6 tests).

- [ ] **Step 7: Run the integration test against a live Ollama (optional)**

Skipped by default. To run manually:

```bash
uv run pytest packages/pipeline/tests/reasoning/test_integration.py -m integration -v
```

Expected: PASS, provided a local Ollama is running with `llama3.2:3b` pulled. The test is forgiving on content but strict on shape.

- [ ] **Step 8: Commit**

```bash
git add packages/pipeline/src/pipeline/reasoning/reasoner.py packages/pipeline/src/pipeline/reasoning/__init__.py packages/pipeline/tests/reasoning/test_reasoner.py packages/pipeline/tests/reasoning/test_integration.py
git commit -m "feat: add generate_reasoning entry point for Phase 5 reasoning"
```

---

## Task 6: Smoke test for the full pipeline as a library

**Files:**
- Modify: `packages/pipeline/tests/test_smoke.py`

The Phase 5 deliverable in the master spec §7 is "full pipeline callable as a library." Add an import-smoke assertion that every stage is importable.

- [ ] **Step 1: Inspect the current smoke test**

Run: `uv run cat packages/pipeline/tests/test_smoke.py` (or open it). Note what the existing test asserts so the addition is non-redundant.

- [ ] **Step 2: Extend (or add) the smoke check**

Modify `packages/pipeline/tests/test_smoke.py` so it includes a test that imports `pipeline.ingestion`, `pipeline.extraction`, `pipeline.scoring`, `pipeline.similarity`, and `pipeline.reasoning`, and confirms each module exposes its primary entry point. Append (do not delete the existing tests):

```python
def test_full_pipeline_is_importable_as_a_library():
    """Phase 5 deliverable: every stage is importable as a pure library."""
    import pipeline.extraction as extraction
    import pipeline.ingestion as ingestion
    import pipeline.reasoning as reasoning
    import pipeline.scoring as scoring
    import pipeline.similarity as similarity

    assert hasattr(ingestion, "ingest_resume_bytes") or hasattr(ingestion, "models")
    assert hasattr(extraction, "extract_resume_profile")
    assert hasattr(scoring, "Scorer")
    assert hasattr(similarity, "SkillMatcher")
    assert hasattr(reasoning, "generate_reasoning")
```

If `pipeline.ingestion` does not expose `ingest_resume_bytes` at the package root, the `hasattr(..., "models")` fallback keeps the test honest without forcing a Phase 1 refactor.

- [ ] **Step 3: Run the smoke test**

Run: `uv run pytest packages/pipeline/tests/test_smoke.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add packages/pipeline/tests/test_smoke.py
git commit -m "test: add smoke test asserting the full pipeline is importable"
```

---

## Task 7: Phase 5 docs and README entry

**Files:**
- Create: `docs/phase-5-reasoning.md`
- Modify: `README.md`

- [ ] **Step 1: Write the Phase 5 user guide**

Create `docs/phase-5-reasoning.md`:

```markdown
# Phase 5 — Reasoning

A pure library that turns the Phase 4 score and skill-match report into 3 reasons and 3 bullet rewrites via a local Ollama LLM.

## What ships

- `pipeline.reasoning.generate_reasoning(...)` — module-level function; one Ollama call returns both reasons and rewrites.
- `pipeline.reasoning.ReasoningResult` / `Reason` / `BulletRewrite` — Pydantic shapes (strict 3 + 3).
- `pipeline.reasoning.ReasoningError` — typed exception for Phase 6 partial-result handling.

`pipeline.reasoning` reuses `pipeline.extraction.client.OllamaClient` as its default backend. See the Phase 5 supplement §2.5 for the rationale.

## Usage

```python
from pipeline.ingestion import ingest_resume_bytes, ingest_job_text
from pipeline.extraction import extract_resume_profile, extract_job_profile
from pipeline.scoring import Scorer
from pipeline.similarity import SkillMatcher
from pipeline.reasoning import generate_reasoning

resume_doc = ingest_resume_bytes(resume_pdf_bytes, filename="resume.pdf")
job_doc = ingest_job_text(jd_text)

resume_profile = extract_resume_profile(resume_doc)
job_profile = extract_job_profile(job_doc)

scorer = Scorer.from_pretrained("USER/resumefit-distilbert-lora", device="cpu")
matcher = SkillMatcher.from_pretrained(device="cpu")

score = scorer.score(resume_doc.raw_text, job_doc.raw_text)
report = matcher.match(resume_profile, job_profile)

reasoning = generate_reasoning(
    score_result=score,
    skill_report=report,
    resume_profile=resume_profile,
    job_profile=job_profile,
    resume_text=resume_doc.raw_text,
)

for r in reasoning.reasons:
    print(f"- [{r.category}] {r.summary} — {r.evidence}")
for b in reasoning.rewrites:
    print(f"  was: {b.original}\n  now: {b.rewritten}\n  why: {b.rationale}\n")
```

## Swapping the model

`generate_reasoning` uses `OllamaClient(model="llama3.2:3b")` by default. To try a stronger model:

```python
from pipeline.extraction.client import OllamaClient

client = OllamaClient(model="qwen2.5:7b")
reasoning = generate_reasoning(..., client=client)
```

## Error handling

`generate_reasoning` raises `ReasoningError` on:

- Ollama transport failure or timeout.
- Unparseable JSON (after `OllamaClient`'s single retry).
- Payload that fails `ReasoningResult` validation (wrong reason/rewrite count, bad category, missing field).

The Phase 6 API catches `ReasoningError` and returns a partial result (score only).

## Testing

Unit tests inject a fake client — no Ollama needed. The integration test is gated:

```bash
# unit tests (default)
uv run pytest packages/pipeline/tests/reasoning -v

# integration test (requires local Ollama with llama3.2:3b)
uv run pytest packages/pipeline/tests/reasoning -m integration -v
```
```

- [ ] **Step 2: Add the Phase 5 entry to the README phases list**

In `README.md`, replace this line:

```markdown
- **Phase 4 — score & similarity:** [plan](docs/superpowers/plans/2026-05-15-phase-4-score.md), [supplement](docs/superpowers/specs/2026-05-15-phase-4-score-supplement.md), [guide](docs/phase-4-score.md).
```

with:

```markdown
- **Phase 4 — score & similarity:** [plan](docs/superpowers/plans/2026-05-15-phase-4-score.md), [supplement](docs/superpowers/specs/2026-05-15-phase-4-score-supplement.md), [guide](docs/phase-4-score.md).
- **Phase 5 — reasoning:** [plan](docs/superpowers/plans/2026-05-15-phase-5-reasoning.md), [supplement](docs/superpowers/specs/2026-05-15-phase-5-reasoning-supplement.md), [guide](docs/phase-5-reasoning.md).
```

- [ ] **Step 3: Run the entire test suite to ensure nothing else regressed**

Run: `uv run pytest -v`
Expected: all tests pass (existing + new Phase 5 tests); integration tests skipped by default.

- [ ] **Step 4: Commit**

```bash
git add docs/phase-5-reasoning.md README.md
git commit -m "docs: add Phase 5 user guide and README entry"
```

---

## Done check

- [ ] All Phase 5 tasks committed.
- [ ] `uv run pytest -v` passes with no failures.
- [ ] `pipeline.reasoning.generate_reasoning`, `ReasoningResult`, and `ReasoningError` are importable from the package root.
- [ ] `Reason.category` enum is enforced in both the Pydantic model and the prompt.
- [ ] The smoke test asserts every stage (`ingestion`, `extraction`, `scoring`, `similarity`, `reasoning`) is importable.
- [ ] No `pipeline.reasoning → apps.api` import was introduced (`grep -R "from apps" packages/pipeline/src` returns nothing).
