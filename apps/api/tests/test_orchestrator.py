from __future__ import annotations

import pytest
from api.dependencies import Pipeline
from api.orchestrator import run_analysis
from api.schemas import AnalyzeResponse
from pipeline.extraction.errors import ExtractionError
from pipeline.ingestion.errors import IngestionError
from pipeline.reasoning.errors import ReasoningError


def _make_pipeline(
    fakes_module,
    *,
    score_result,
    skill_report,
    resume_profile,
    job_profile,
    reasoning_result,
    extract_resume_exc=None,
    reasoning_exc=None,
) -> Pipeline:
    extract_resume_fn = (
        fakes_module.failing_extract_resume(extract_resume_exc)
        if extract_resume_exc is not None
        else fakes_module.ok_extract_resume(resume_profile)
    )
    reasoning_fn = (
        fakes_module.failing_reasoning(reasoning_exc)
        if reasoning_exc is not None
        else fakes_module.ok_reasoning(reasoning_result)
    )
    return Pipeline(
        scorer=fakes_module.FakeScorer(next_result=score_result),
        matcher=fakes_module.FakeMatcher(next_report=skill_report),
        ollama_client=fakes_module.FakeOllama(),
        extract_resume_fn=extract_resume_fn,
        extract_job_fn=fakes_module.ok_extract_job(job_profile),
        reasoning_fn=reasoning_fn,
    )


def test_run_analysis_happy_path(
    fakes_module,
    resume_txt_bytes,
    jd_text,
    sample_score_result,
    sample_skill_report,
    sample_resume_profile,
    sample_job_profile,
    sample_reasoning_result,
):
    pipeline = _make_pipeline(
        fakes_module,
        score_result=sample_score_result,
        skill_report=sample_skill_report,
        resume_profile=sample_resume_profile,
        job_profile=sample_job_profile,
        reasoning_result=sample_reasoning_result,
    )
    response = run_analysis(
        resume_bytes=resume_txt_bytes,
        filename="jane.txt",
        jd_text=jd_text,
        pipeline=pipeline,
    )
    assert isinstance(response, AnalyzeResponse)
    assert response.score == sample_score_result
    assert response.skill_report == sample_skill_report
    assert response.reasoning == sample_reasoning_result
    assert response.warnings == []


def test_run_analysis_raises_ingestion_error_on_bad_pdf(
    fakes_module,
    jd_text,
    sample_score_result,
    sample_skill_report,
    sample_resume_profile,
    sample_job_profile,
    sample_reasoning_result,
):
    pipeline = _make_pipeline(
        fakes_module,
        score_result=sample_score_result,
        skill_report=sample_skill_report,
        resume_profile=sample_resume_profile,
        job_profile=sample_job_profile,
        reasoning_result=sample_reasoning_result,
    )
    with pytest.raises(IngestionError):
        run_analysis(
            resume_bytes=b"not actually a pdf",
            filename="jane.pdf",
            jd_text=jd_text,
            pipeline=pipeline,
        )


def test_run_analysis_raises_ingestion_error_on_empty_jd(
    fakes_module,
    resume_txt_bytes,
    sample_score_result,
    sample_skill_report,
    sample_resume_profile,
    sample_job_profile,
    sample_reasoning_result,
):
    pipeline = _make_pipeline(
        fakes_module,
        score_result=sample_score_result,
        skill_report=sample_skill_report,
        resume_profile=sample_resume_profile,
        job_profile=sample_job_profile,
        reasoning_result=sample_reasoning_result,
    )
    with pytest.raises(IngestionError):
        run_analysis(
            resume_bytes=resume_txt_bytes,
            filename="jane.txt",
            jd_text="   \n\t  ",
            pipeline=pipeline,
        )


def test_run_analysis_degrades_to_score_only_when_extraction_fails(
    fakes_module,
    resume_txt_bytes,
    jd_text,
    sample_score_result,
    sample_skill_report,
    sample_resume_profile,
    sample_job_profile,
    sample_reasoning_result,
):
    pipeline = _make_pipeline(
        fakes_module,
        score_result=sample_score_result,
        skill_report=sample_skill_report,
        resume_profile=sample_resume_profile,
        job_profile=sample_job_profile,
        reasoning_result=sample_reasoning_result,
        extract_resume_exc=ExtractionError("Ollama unreachable: connection refused"),
    )
    response = run_analysis(
        resume_bytes=resume_txt_bytes,
        filename="jane.txt",
        jd_text=jd_text,
        pipeline=pipeline,
    )
    assert response.score == sample_score_result
    assert response.skill_report is None
    assert response.reasoning is None
    assert len(response.warnings) == 1
    assert "extraction" in response.warnings[0].lower()
    assert "Ollama unreachable" in response.warnings[0]


def test_run_analysis_degrades_reasoning_only_when_reasoning_fails(
    fakes_module,
    resume_txt_bytes,
    jd_text,
    sample_score_result,
    sample_skill_report,
    sample_resume_profile,
    sample_job_profile,
    sample_reasoning_result,
):
    pipeline = _make_pipeline(
        fakes_module,
        score_result=sample_score_result,
        skill_report=sample_skill_report,
        resume_profile=sample_resume_profile,
        job_profile=sample_job_profile,
        reasoning_result=sample_reasoning_result,
        reasoning_exc=ReasoningError("Ollama returned invalid ReasoningResult"),
    )
    response = run_analysis(
        resume_bytes=resume_txt_bytes,
        filename="jane.txt",
        jd_text=jd_text,
        pipeline=pipeline,
    )
    assert response.score == sample_score_result
    assert response.skill_report == sample_skill_report
    assert response.reasoning is None
    assert len(response.warnings) == 1
    assert "reasoning" in response.warnings[0].lower()
    assert "invalid" in response.warnings[0]


def test_run_analysis_threads_ollama_client_into_extract_and_reasoning(
    fakes_module,
    resume_txt_bytes,
    jd_text,
    sample_score_result,
    sample_skill_report,
    sample_resume_profile,
    sample_job_profile,
    sample_reasoning_result,
):
    seen: dict[str, object] = {}

    def extract_resume_fn(doc, **kw):
        seen["extract_resume_client"] = kw.get("client")
        return sample_resume_profile

    def extract_job_fn(doc, **kw):
        seen["extract_job_client"] = kw.get("client")
        return sample_job_profile

    def reasoning_fn(**kw):
        seen["reasoning_client"] = kw.get("client")
        return sample_reasoning_result

    fake_ollama = fakes_module.FakeOllama()
    pipeline = Pipeline(
        scorer=fakes_module.FakeScorer(next_result=sample_score_result),
        matcher=fakes_module.FakeMatcher(next_report=sample_skill_report),
        ollama_client=fake_ollama,
        extract_resume_fn=extract_resume_fn,
        extract_job_fn=extract_job_fn,
        reasoning_fn=reasoning_fn,
    )
    run_analysis(
        resume_bytes=resume_txt_bytes,
        filename="jane.txt",
        jd_text=jd_text,
        pipeline=pipeline,
    )
    assert seen["extract_resume_client"] is fake_ollama
    assert seen["extract_job_client"] is fake_ollama
    assert seen["reasoning_client"] is fake_ollama
