from __future__ import annotations

from pipeline.extraction.models import JobProfile, ResumeProfile
from pipeline.scoring.models import ScoreResult
from pipeline.similarity.models import SkillMatch, SkillMatchReport

MAX_RESUME_CHARS = 6000


_TEMPLATE = """You are an expert technical recruiter explaining why a resume fits a job and suggesting concrete improvements.

You will receive (1) a fit score with confidence, (2) a skill match report, (3) the candidate's profile, (4) the job's profile, and (5) the candidate's resume text.

Return ONLY a JSON object with this exact shape — no prose, no markdown:

{{
  "reasons": [
    {{"summary": "string", "evidence": "string", "category": "matched_skill|missing_skill|experience_match|experience_gap|other"}},
    {{"summary": "string", "evidence": "string", "category": "matched_skill|missing_skill|experience_match|experience_gap|other"}},
    {{"summary": "string", "evidence": "string", "category": "matched_skill|missing_skill|experience_match|experience_gap|other"}}
  ],
  "rewrites": [
    {{"original": "string", "rewritten": "string", "rationale": "string"}},
    {{"original": "string", "rewritten": "string", "rationale": "string"}},
    {{"original": "string", "rewritten": "string", "rationale": "string"}}
  ]
}}

Rules:
- Return exactly 3 reasons and exactly 3 rewrites — no more, no fewer.
- "category" must be one of: matched_skill, missing_skill, experience_match, experience_gap, other.
- Reasons should be specific (cite concrete skills, years, or roles) — never generic platitudes.
- Each rewrite picks an existing resume bullet ("original") and improves it for this specific JD ("rewritten"), then explains why ("rationale"). If no bullet fits, set "original" to "" and synthesize a new bullet.
- Rewritten bullets should lead with an action verb, include a metric where plausible, and reference a skill or technology the JD requires.

Fit score: {score:.1f}/100  (predicted: {label}, confidence: {confidence:.2f})
Required match rate: {match_rate:.0%}

Matched required skills: {req_matched}
Missing required skills: {req_missing}
Matched nice-to-have skills: {nice_matched}
Missing nice-to-have skills: {nice_missing}

Job:
- title: {job_title}
- seniority: {seniority}
- min years experience: {min_years}
- required skills: {required_skills}
- nice-to-have skills: {nice_skills}

Candidate profile:
- titles: {resume_titles}
- skills: {resume_skills}
- total years experience: {total_years}
- education: {education}

Resume text:
---
{resume_text}
---
"""


def _skill_list(matches: list[SkillMatch]) -> str:
    if not matches:
        return "(none)"
    return ", ".join(m.jd_skill for m in matches)


def build_reasoning_prompt(
    *,
    score_result: ScoreResult,
    skill_report: SkillMatchReport,
    resume_profile: ResumeProfile,
    job_profile: JobProfile,
    resume_text: str,
) -> str:
    """Render the single prompt that asks the LLM for 3 reasons + 3 rewrites."""
    truncated_resume = resume_text[:MAX_RESUME_CHARS]
    return _TEMPLATE.format(
        score=score_result.score,
        label=score_result.predicted_label,
        confidence=score_result.confidence,
        match_rate=skill_report.match_rate,
        req_matched=_skill_list(skill_report.required_matched),
        req_missing=_skill_list(skill_report.required_missing),
        nice_matched=_skill_list(skill_report.nice_to_have_matched),
        nice_missing=_skill_list(skill_report.nice_to_have_missing),
        job_title=job_profile.title,
        seniority=job_profile.seniority,
        min_years=job_profile.min_years_experience,
        required_skills=", ".join(job_profile.required_skills) or "(none)",
        nice_skills=", ".join(job_profile.nice_to_have_skills) or "(none)",
        resume_titles=", ".join(resume_profile.titles) or "(none)",
        resume_skills=", ".join(resume_profile.skills) or "(none)",
        total_years=resume_profile.total_years_experience,
        education=", ".join(resume_profile.education) or "(none)",
        resume_text=truncated_resume,
    )
