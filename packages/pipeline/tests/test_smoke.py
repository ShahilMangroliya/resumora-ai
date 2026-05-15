import pipeline


def test_pipeline_is_importable():
    assert pipeline.__version__ == "0.1.0"


def test_full_pipeline_is_importable_as_a_library():
    """Phase 5 deliverable: every stage is importable as a pure library."""
    import pipeline.extraction as extraction
    import pipeline.ingestion as ingestion
    import pipeline.reasoning as reasoning
    import pipeline.scoring as scoring
    import pipeline.similarity as similarity

    assert hasattr(ingestion, "ingest_resume")
    assert hasattr(ingestion, "ingest_job")
    assert hasattr(extraction, "extract_resume_profile")
    assert hasattr(extraction, "extract_job_profile")
    assert hasattr(scoring, "Scorer")
    assert hasattr(similarity, "SkillMatcher")
    assert hasattr(reasoning, "generate_reasoning")
    assert hasattr(reasoning, "ReasoningError")
