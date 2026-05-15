from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pipeline.ingestion.errors import IngestionError

from api.config import load_settings
from api.dependencies import (
    Pipeline,
    get_matcher,
    get_ollama_client,
    get_pipeline,
    get_scorer,
)
from api.orchestrator import run_analysis
from api.schemas import AnalyzeResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = load_settings()
    if settings.warmup_on_startup:
        # Eagerly construct singletons so a bad model repo / unreachable
        # Ollama surfaces at startup instead of on the first request.
        get_scorer()
        get_matcher()
        get_ollama_client()
    yield


app = FastAPI(title="ResumeFit API", lifespan=lifespan)


_settings = load_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_settings.cors_origins),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(IngestionError)
async def _ingestion_error_handler(request: Request, exc: IngestionError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    resume: UploadFile,
    job_description: str = Form(...),
    pipeline: Pipeline = Depends(get_pipeline),
) -> AnalyzeResponse:
    """Score a resume against a job description and explain the fit.

    Returns 200 with a (possibly partial) `AnalyzeResponse`. Returns 400 if
    the resume file cannot be parsed or the JD is empty.
    """
    if resume.filename is None:
        raise HTTPException(status_code=400, detail="resume upload missing a filename")
    resume_bytes = resume.file.read()
    return run_analysis(
        resume_bytes=resume_bytes,
        filename=resume.filename,
        jd_text=job_description,
        pipeline=pipeline,
    )
