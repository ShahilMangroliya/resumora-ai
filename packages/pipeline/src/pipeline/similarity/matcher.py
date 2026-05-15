from __future__ import annotations

from typing import Self

import numpy as np

from pipeline.extraction.models import JobProfile, ResumeProfile
from pipeline.similarity._embeddings import EmbeddingBackend, SentenceTransformerBackend
from pipeline.similarity.models import SkillMatch, SkillMatchReport

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_THRESHOLD = 0.55


def _normalize(skill: str) -> str:
    return skill.lower().strip()


class SkillMatcher:
    """Embedding-based skill matcher.

    For every JD skill, the closest resume skill (by cosine similarity) is the
    candidate. If the similarity meets `threshold`, the pair lands in the
    matched list; otherwise it lands in the missing list (still carrying the
    closest-but-rejected resume skill, for downstream messaging).
    """

    def __init__(self, *, backend: EmbeddingBackend, threshold: float = _DEFAULT_THRESHOLD) -> None:
        self._backend = backend
        self._threshold = threshold

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = _DEFAULT_MODEL,
        *,
        device: str = "cpu",
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> Self:
        backend = SentenceTransformerBackend(model_name, device=device)
        return cls(backend=backend, threshold=threshold)

    def match(self, resume: ResumeProfile, job: JobProfile) -> SkillMatchReport:
        required_matched, required_missing = self._match_list(
            resume_skills=resume.skills, jd_skills=job.required_skills
        )
        nice_matched, nice_missing = self._match_list(
            resume_skills=resume.skills, jd_skills=job.nice_to_have_skills
        )
        total_required = len(job.required_skills)
        match_rate = 1.0 if total_required == 0 else len(required_matched) / total_required
        return SkillMatchReport(
            required_matched=required_matched,
            required_missing=required_missing,
            nice_to_have_matched=nice_matched,
            nice_to_have_missing=nice_missing,
            match_rate=match_rate,
        )

    def _match_list(
        self,
        *,
        resume_skills: list[str],
        jd_skills: list[str],
    ) -> tuple[list[SkillMatch], list[SkillMatch]]:
        matched: list[SkillMatch] = []
        missing: list[SkillMatch] = []
        if not jd_skills:
            return matched, missing

        if not resume_skills:
            for jd_skill in jd_skills:
                missing.append(
                    SkillMatch(jd_skill=jd_skill, resume_skill="", similarity=0.0, matched=False)
                )
            return matched, missing

        jd_norm = [_normalize(s) for s in jd_skills]
        rs_norm = [_normalize(s) for s in resume_skills]
        jd_vecs = self._backend.encode(jd_norm)
        rs_vecs = self._backend.encode(rs_norm)

        # Cosine on unit-normalized vectors is the dot product.
        sims = jd_vecs @ rs_vecs.T  # shape (J, R)

        for i, jd_skill in enumerate(jd_skills):
            row = sims[i]
            best_idx = int(np.argmax(row))
            best_sim = float(row[best_idx])
            # Clamp into [0, 1] — sentence-transformer cosine can dip very slightly negative.
            best_sim = max(0.0, min(1.0, best_sim))
            entry = SkillMatch(
                jd_skill=jd_skill,
                resume_skill=resume_skills[best_idx],
                similarity=best_sim,
                matched=best_sim >= self._threshold,
            )
            (matched if entry.matched else missing).append(entry)
        return matched, missing
