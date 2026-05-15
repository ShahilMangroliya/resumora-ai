# Phase 5 — Reasoning

A pure library that turns the Phase 4 score and skill-match report into 3 reasons and 3 bullet rewrites via a local Ollama LLM.

## What ships

- `pipeline.reasoning.generate_reasoning(...)` — module-level function; one Ollama call returns both reasons and rewrites.
- `pipeline.reasoning.ReasoningResult` / `Reason` / `BulletRewrite` — Pydantic shapes (strict 3 + 3).
- `pipeline.reasoning.ReasoningError` — typed exception for Phase 6 partial-result handling.

`pipeline.reasoning` reuses `pipeline.extraction.client.OllamaClient` as its default backend. See the Phase 5 supplement §2.5 for the rationale.

## Usage

```python
from pipeline.ingestion import ingest_resume, ingest_job
from pipeline.extraction import extract_resume_profile, extract_job_profile
from pipeline.scoring import Scorer
from pipeline.similarity import SkillMatcher
from pipeline.reasoning import generate_reasoning

resume_doc = ingest_resume(resume_pdf_bytes, filename="resume.pdf")
job_doc = ingest_job(jd_text)

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
