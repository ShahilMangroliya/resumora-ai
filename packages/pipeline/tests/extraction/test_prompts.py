from pipeline.extraction.prompts import build_job_prompt, build_resume_prompt


def test_resume_prompt_includes_the_source_text():
    text = "Jane Doe — Senior Engineer"
    prompt = build_resume_prompt(text)
    assert text in prompt
    assert "JSON" in prompt
    assert "titles" in prompt
    assert "skills" in prompt
    assert "total_years_experience" in prompt


def test_job_prompt_includes_the_source_text():
    text = "We are hiring a backend engineer."
    prompt = build_job_prompt(text)
    assert text in prompt
    assert "JSON" in prompt
    assert "required_skills" in prompt
    assert "seniority" in prompt


def test_resume_prompt_is_deterministic():
    a = build_resume_prompt("x")
    b = build_resume_prompt("x")
    assert a == b
