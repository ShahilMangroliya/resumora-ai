RESUME_PROMPT = """You are extracting structured fields from a resume.

Return ONLY a JSON object with this exact shape — no prose, no markdown:

{{
  "titles": ["string"],
  "skills": ["string"],
  "experiences": [{{"title": "string", "years": 0.0}}],
  "education": ["string"],
  "total_years_experience": 0.0
}}

Rules:
- skills are lowercase short tokens (e.g. "python", "fastapi", "docker").
- experiences.years is the time spent in that role (decimal years).
- total_years_experience is the sum across non-overlapping roles.
- If a field cannot be determined, return an empty list or 0.

Resume:
---
{text}
---
"""


JOB_PROMPT = """You are extracting structured fields from a job description.

Return ONLY a JSON object with this exact shape — no prose, no markdown:

{{
  "title": "string",
  "required_skills": ["string"],
  "nice_to_have_skills": ["string"],
  "seniority": "intern|junior|mid|senior|staff|principal",
  "min_years_experience": 0.0
}}

Rules:
- skills are lowercase short tokens.
- seniority must be one of the listed values; pick the closest match.
- min_years_experience is a decimal; use 0 if unspecified.

Job description:
---
{text}
---
"""


def build_resume_prompt(text: str) -> str:
    """Render the resume-extraction prompt for a given resume text."""
    return RESUME_PROMPT.format(text=text)


def build_job_prompt(text: str) -> str:
    """Render the job-extraction prompt for a given JD text."""
    return JOB_PROMPT.format(text=text)
