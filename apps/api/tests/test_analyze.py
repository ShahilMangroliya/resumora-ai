from __future__ import annotations

import io

import pytest
from api.dependencies import Pipeline, get_pipeline
from api.main import app
from fastapi.testclient import TestClient
from pipeline.extraction.errors import ExtractionError
from pipeline.reasoning.errors import ReasoningError


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _override_pipeline(
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
    pipeline = Pipeline(
        scorer=fakes_module.FakeScorer(next_result=score_result),
        matcher=fakes_module.FakeMatcher(next_report=skill_report),
        ollama_client=fakes_module.FakeOllama(),
        extract_resume_fn=extract_resume_fn,
        extract_job_fn=fakes_module.ok_extract_job(job_profile),
        reasoning_fn=reasoning_fn,
    )
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    return pipeline


def test_analyze_happy_path_returns_full_response(
    client,
    fakes_module,
    resume_txt_bytes,
    jd_text,
    sample_score_result,
    sample_skill_report,
    sample_resume_profile,
    sample_job_profile,
    sample_reasoning_result,
):
    _override_pipeline(
        fakes_module,
        score_result=sample_score_result,
        skill_report=sample_skill_report,
        resume_profile=sample_resume_profile,
        job_profile=sample_job_profile,
        reasoning_result=sample_reasoning_result,
    )
    response = client.post(
        "/analyze",
        files={"resume": ("jane.txt", io.BytesIO(resume_txt_bytes), "text/plain")},
        data={"job_description": jd_text},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["score"]["predicted_label"] == "strong"
    assert body["skill_report"] is not None
    assert body["reasoning"] is not None
    assert len(body["reasoning"]["reasons"]) == 3
    assert len(body["reasoning"]["rewrites"]) == 3
    assert body["warnings"] == []


def test_analyze_rejects_unsupported_extension(
    client,
    fakes_module,
    resume_txt_bytes,
    jd_text,
    sample_score_result,
    sample_skill_report,
    sample_resume_profile,
    sample_job_profile,
    sample_reasoning_result,
):
    _override_pipeline(
        fakes_module,
        score_result=sample_score_result,
        skill_report=sample_skill_report,
        resume_profile=sample_resume_profile,
        job_profile=sample_job_profile,
        reasoning_result=sample_reasoning_result,
    )
    response = client.post(
        "/analyze",
        files={"resume": ("jane.rtf", io.BytesIO(resume_txt_bytes), "application/rtf")},
        data={"job_description": jd_text},
    )
    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]


def test_analyze_rejects_empty_jd(
    client,
    fakes_module,
    resume_txt_bytes,
    sample_score_result,
    sample_skill_report,
    sample_resume_profile,
    sample_job_profile,
    sample_reasoning_result,
):
    _override_pipeline(
        fakes_module,
        score_result=sample_score_result,
        skill_report=sample_skill_report,
        resume_profile=sample_resume_profile,
        job_profile=sample_job_profile,
        reasoning_result=sample_reasoning_result,
    )
    response = client.post(
        "/analyze",
        files={"resume": ("jane.txt", io.BytesIO(resume_txt_bytes), "text/plain")},
        data={"job_description": "   \t\n "},
    )
    assert response.status_code == 400
    assert "no readable text" in response.json()["detail"]


def test_analyze_returns_partial_when_extraction_fails(
    client,
    fakes_module,
    resume_txt_bytes,
    jd_text,
    sample_score_result,
    sample_skill_report,
    sample_resume_profile,
    sample_job_profile,
    sample_reasoning_result,
):
    _override_pipeline(
        fakes_module,
        score_result=sample_score_result,
        skill_report=sample_skill_report,
        resume_profile=sample_resume_profile,
        job_profile=sample_job_profile,
        reasoning_result=sample_reasoning_result,
        extract_resume_exc=ExtractionError("Ollama unreachable: connection refused"),
    )
    response = client.post(
        "/analyze",
        files={"resume": ("jane.txt", io.BytesIO(resume_txt_bytes), "text/plain")},
        data={"job_description": jd_text},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["score"]["predicted_label"] == "strong"
    assert body["skill_report"] is None
    assert body["reasoning"] is None
    assert len(body["warnings"]) == 1
    assert "extraction" in body["warnings"][0].lower()


def test_analyze_returns_partial_when_reasoning_fails(
    client,
    fakes_module,
    resume_txt_bytes,
    jd_text,
    sample_score_result,
    sample_skill_report,
    sample_resume_profile,
    sample_job_profile,
    sample_reasoning_result,
):
    _override_pipeline(
        fakes_module,
        score_result=sample_score_result,
        skill_report=sample_skill_report,
        resume_profile=sample_resume_profile,
        job_profile=sample_job_profile,
        reasoning_result=sample_reasoning_result,
        reasoning_exc=ReasoningError("Ollama returned invalid ReasoningResult"),
    )
    response = client.post(
        "/analyze",
        files={"resume": ("jane.txt", io.BytesIO(resume_txt_bytes), "text/plain")},
        data={"job_description": jd_text},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["score"]["predicted_label"] == "strong"
    assert body["skill_report"] is not None
    assert body["reasoning"] is None
    assert len(body["warnings"]) == 1
    assert "reasoning" in body["warnings"][0].lower()


def test_analyze_requires_resume_file(client):
    response = client.post("/analyze", data={"job_description": "Some JD text."})
    assert response.status_code == 422  # FastAPI validation: missing required form field


def test_analyze_requires_job_description(
    client,
    fakes_module,
    resume_txt_bytes,
    sample_score_result,
    sample_skill_report,
    sample_resume_profile,
    sample_job_profile,
    sample_reasoning_result,
):
    _override_pipeline(
        fakes_module,
        score_result=sample_score_result,
        skill_report=sample_skill_report,
        resume_profile=sample_resume_profile,
        job_profile=sample_job_profile,
        reasoning_result=sample_reasoning_result,
    )
    response = client.post(
        "/analyze",
        files={"resume": ("jane.txt", io.BytesIO(resume_txt_bytes), "text/plain")},
    )
    assert response.status_code == 422
