# Phase 4 — Score & Similarity

Two pure libraries that turn a `(resume, job description)` pair into a fit score and a matched / missing skills report.

## What ships

- `pipeline.scoring.Scorer` — loads the fine-tuned Phase 3 DistilBERT+LoRA model from HF Hub (or a local directory) and scores a pair.
- `pipeline.similarity.SkillMatcher` — embeds skill phrases with `sentence-transformers/all-MiniLM-L6-v2` and returns a `SkillMatchReport`.

Neither module depends on FastAPI, Ollama, or `packages/training`. They are imported by the Phase 6 API.

## Score range is `[20, 85]`

The scorer's output is bounded to `[20.0, 85.0]` — softmax · `[20, 55, 85]`. This is a deliberate Phase 3 design decision (see `docs/superpowers/specs/2026-05-15-phase-3-finetune-supplement.md` §1.1). The product surface ("0–100 fit score") is honored by disclosure, not by rescaling.

## Usage

```python
from pipeline.scoring import Scorer
from pipeline.similarity import SkillMatcher
from pipeline.extraction import extract_resume_profile, extract_job_profile
from pipeline.ingestion import ingest_resume_bytes, ingest_job_text

scorer = Scorer.from_pretrained("USER/resumefit-distilbert-lora", device="cpu")
matcher = SkillMatcher.from_pretrained(device="cpu")  # MiniLM by default

resume_doc = ingest_resume_bytes(resume_pdf_bytes, filename="resume.pdf")
job_doc = ingest_job_text(jd_text)

resume_profile = extract_resume_profile(resume_doc)
job_profile = extract_job_profile(job_doc)

result = scorer.score(resume_doc.raw_text, job_doc.raw_text)
report = matcher.match(resume_profile, job_profile)

print(f"Score: {result.score:.1f} (confidence {result.confidence:.2f})")
print(f"Required match rate: {report.match_rate:.0%}")
for miss in report.required_missing:
    print(f"  missing: {miss.jd_skill}  (closest: {miss.resume_skill}, {miss.similarity:.2f})")
```

## Loading from a local adapter

For development against a smoke-trained adapter:

```python
scorer = Scorer.from_pretrained(
    repo_id_or_path="outputs/smoke",   # local PEFT adapter dir
    base_model="distilbert-base-uncased",
    device="cpu",
)
```

## Tuning the similarity threshold

The default threshold is `0.55` (cosine on MiniLM normalized embeddings). Lower to admit more matches; raise to be stricter:

```python
matcher = SkillMatcher.from_pretrained(threshold=0.6)
```

## Testing

Unit tests do not download models: scoring uses a config-built tiny DistilBERT; similarity uses an injected fake `EmbeddingBackend`. Integration tests (real Hub model, real sentence-transformer) are gated:

```bash
# unit tests (default)
uv run pytest packages/pipeline/tests/scoring packages/pipeline/tests/similarity -v

# integration tests
uv run pytest packages/pipeline/tests/scoring packages/pipeline/tests/similarity -m integration -v
```

The Hub integration test needs `RESUMEFIT_SCORER_REPO` pointing at the published Phase 3 model:

```bash
RESUMEFIT_SCORER_REPO=USER/resumefit-distilbert-lora \
  uv run pytest packages/pipeline/tests/scoring -m integration -v
```
