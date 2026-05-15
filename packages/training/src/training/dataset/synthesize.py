from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Protocol

from pydantic import ValidationError

from training.dataset.roles import RoleSpec
from training.dataset.schema import Label, Pair, score_for_label


_PROMPT = """You are generating one synthetic training example for a resume / job-fit classifier.

Generate ONE (resume, job description) PAIR where the resume is a {label_phrase} fit for the JD.

Role: {role_title} ({family})
Seniority: {seniority}
Skills the role usually expects: {skills}
Variation seed: {seed}

Definitions:
- "strong" — the resume clearly meets or exceeds the JD's requirements.
- "partial" — the resume has some of the required skills but is missing important ones or is one seniority level off.
- "weak" — the resume is in a different domain or far below the requirements.

Return ONLY a JSON object with this exact shape — no prose, no markdown:

{{
  "resume_text": "a realistic short resume (8-20 lines): name, headline, 1-3 jobs with dates, 1-2 education lines",
  "jd_text": "a realistic short job description (8-20 lines): role title, 4-8 bullet requirements, 2-4 nice-to-have bullets"
}}

Vary names, companies, formatting between examples.
"""

_LABEL_PHRASE = {"strong": "STRONG", "partial": "PARTIAL", "weak": "WEAK"}


class _Client(Protocol):
    def generate_json(self, prompt: str) -> dict: ...


def _pair_id(role: RoleSpec, seniority: str, label: Label, seed: int) -> str:
    raw = f"{role.slug}|{seniority}|{label}|{seed}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_pair(
    role: RoleSpec,
    seniority: str,
    label: Label,
    *,
    client: _Client,
    seed: int,
    model_name: str = "llama3.2:3b",
) -> Pair:
    """Generate one (resume, JD) Pair for a (role, seniority, label) triple."""
    prompt = _PROMPT.format(
        label_phrase=_LABEL_PHRASE[label],
        role_title=role.title,
        family=role.family,
        seniority=seniority,
        skills=", ".join(role.core_skills),
        seed=seed,
    )
    payload = client.generate_json(prompt)
    try:
        return Pair(
            pair_id=_pair_id(role, seniority, label, seed),
            resume_text=payload["resume_text"],
            jd_text=payload["jd_text"],
            label=label,
            score=score_for_label(label),
            role=role.slug,
            seniority=seniority,
            source="synthetic",
            generator_model=model_name,
            generated_at=_now_utc_iso(),
            prompt_seed=seed,
        )
    except (KeyError, ValidationError) as exc:
        raise ValueError(f"Ollama returned malformed pair payload: {exc}") from exc
