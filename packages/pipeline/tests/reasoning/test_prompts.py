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
            SkillMatch(
                jd_skill="kubernetes", resume_skill="docker", similarity=0.42, matched=False
            ),
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
    assert "5" in prompt


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
    assert "3" in prompt or "three" in prompt.lower()
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
    for cat in (
        "matched_skill",
        "missing_skill",
        "experience_match",
        "experience_gap",
        "other",
    ):
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
    prompt = build_reasoning_prompt(
        score_result=_score_result(),
        skill_report=empty_report,
        resume_profile=_resume_profile(),
        job_profile=empty_job,
        resume_text="x",
    )
    assert isinstance(prompt, str)
    assert len(prompt) > 0
