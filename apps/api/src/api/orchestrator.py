from __future__ import annotations

from pipeline.extraction.errors import ExtractionError
from pipeline.ingestion import ingest_job, ingest_resume
from pipeline.reasoning.errors import ReasoningError

from api.dependencies import Pipeline
from api.schemas import AnalyzeResponse


def run_analysis(
    *,
    resume_bytes: bytes,
    filename: str,
    jd_text: str,
    pipeline: Pipeline,
) -> AnalyzeResponse:
    """Run the five-stage Resumora AI pipeline synchronously.

    Raises:
        IngestionError: when the resume or JD cannot be ingested. The caller
            (FastAPI route) maps this to HTTP 400.

    Returns:
        AnalyzeResponse. Partial when extraction or reasoning fails — the
        `warnings` list explains which stages were skipped.
    """
    resume_doc = ingest_resume(resume_bytes, filename)
    job_doc = ingest_job(jd_text)

    warnings: list[str] = []

    resume_profile = None
    job_profile = None
    try:
        resume_profile = pipeline.extract_resume_fn(resume_doc, client=pipeline.ollama_client)
        job_profile = pipeline.extract_job_fn(job_doc, client=pipeline.ollama_client)
    except ExtractionError as exc:
        warnings.append(f"Profile extraction failed; downstream stages skipped: {exc}")
        resume_profile = None
        job_profile = None

    score = pipeline.scorer.score(resume_doc.raw_text, job_doc.raw_text)

    skill_report = None
    if resume_profile is not None and job_profile is not None:
        skill_report = pipeline.matcher.match(resume_profile, job_profile)

    reasoning = None
    if (
        resume_profile is not None
        and job_profile is not None
        and skill_report is not None
    ):
        try:
            reasoning = pipeline.reasoning_fn(
                score_result=score,
                skill_report=skill_report,
                resume_profile=resume_profile,
                job_profile=job_profile,
                resume_text=resume_doc.raw_text,
                client=pipeline.ollama_client,
            )
        except ReasoningError as exc:
            warnings.append(f"Reasoning generation failed: {exc}")

    return AnalyzeResponse(
        score=score,
        skill_report=skill_report,
        reasoning=reasoning,
        warnings=warnings,
    )
