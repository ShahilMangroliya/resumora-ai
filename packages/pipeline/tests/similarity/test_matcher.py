import math

import numpy as np

from pipeline.extraction.models import JobProfile, ResumeProfile
from pipeline.similarity import SkillMatch, SkillMatcher, SkillMatchReport


class _FixedBackend:
    """Returns hand-set unit vectors for a known vocabulary (after lower/strip)."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vecs: dict[str, np.ndarray] = {}
        for k, v in vectors.items():
            arr = np.array(v, dtype=np.float64)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            self._vecs[k] = arr

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vecs[t] for t in texts])


def _resume(skills: list[str]) -> ResumeProfile:
    return ResumeProfile(
        titles=["engineer"],
        skills=skills,
        experiences=[],
        education=[],
        total_years_experience=3.0,
    )


def _job(required: list[str], nice: list[str]) -> JobProfile:
    return JobProfile(
        title="engineer",
        required_skills=required,
        nice_to_have_skills=nice,
        seniority="mid",
        min_years_experience=2.0,
    )


def test_match_above_threshold_counts_as_matched():
    backend = _FixedBackend({"python": [1, 0], "python3": [0.95, 0.3]})
    matcher = SkillMatcher(backend=backend, threshold=0.55)
    report = matcher.match(_resume(["Python3"]), _job(["Python"], []))
    assert len(report.required_matched) == 1
    assert report.required_matched[0].jd_skill == "Python"
    assert report.required_matched[0].resume_skill == "Python3"
    assert report.required_matched[0].matched is True
    assert report.required_missing == []


def test_match_below_threshold_counts_as_missing():
    backend = _FixedBackend({"python": [1, 0], "java": [0, 1]})
    matcher = SkillMatcher(backend=backend, threshold=0.55)
    report = matcher.match(_resume(["Java"]), _job(["Python"], []))
    assert report.required_matched == []
    assert len(report.required_missing) == 1
    miss = report.required_missing[0]
    assert miss.jd_skill == "Python"
    assert miss.resume_skill == "Java"
    assert miss.matched is False


def test_match_rate_required_only():
    backend = _FixedBackend({
        "python": [1, 0, 0],
        "django": [0, 1, 0],
        "rust": [0, 0, 1],
    })
    matcher = SkillMatcher(backend=backend, threshold=0.55)
    report = matcher.match(_resume(["Python"]), _job(["Python", "Django"], []))
    assert report.match_rate == 0.5


def test_empty_resume_skills_means_everything_missing():
    backend = _FixedBackend({"python": [1, 0]})
    matcher = SkillMatcher(backend=backend, threshold=0.55)
    report = matcher.match(_resume([]), _job(["Python"], []))
    assert report.required_matched == []
    assert len(report.required_missing) == 1
    miss = report.required_missing[0]
    assert miss.resume_skill == ""
    assert miss.similarity == 0.0
    assert report.match_rate == 0.0


def test_empty_required_skills_match_rate_one():
    backend = _FixedBackend({"python": [1, 0]})
    matcher = SkillMatcher(backend=backend, threshold=0.55)
    report = matcher.match(_resume(["Python"]), _job([], []))
    assert report.match_rate == 1.0
    assert report.required_matched == []
    assert report.required_missing == []


def test_nice_to_have_uses_same_threshold():
    backend = _FixedBackend({"docker": [1, 0], "kubernetes": [0, 1]})
    matcher = SkillMatcher(backend=backend, threshold=0.55)
    report = matcher.match(_resume(["Docker"]), _job([], ["Docker", "Kubernetes"]))
    assert len(report.nice_to_have_matched) == 1
    assert len(report.nice_to_have_missing) == 1
    # match_rate only reflects required.
    assert report.match_rate == 1.0


def test_normalization_lowers_and_strips():
    # "PYTHON " and " python" must hit the same vector via the normalize step.
    backend = _FixedBackend({"python": [1, 0]})
    matcher = SkillMatcher(backend=backend, threshold=0.55)
    report = matcher.match(_resume(["PYTHON "]), _job([" python"], []))
    assert len(report.required_matched) == 1
    assert report.required_matched[0].similarity > 0.99


def test_threshold_is_inclusive_at_boundary():
    # Build vectors at exact similarity = 0.55.
    a = [1.0, 0.0]
    b = [0.55, math.sqrt(1 - 0.55 ** 2)]
    backend = _FixedBackend({"a": a, "b": b})
    matcher = SkillMatcher(backend=backend, threshold=0.55)
    report = matcher.match(_resume(["A"]), _job(["B"], []))
    assert len(report.required_matched) == 1
    assert report.required_matched[0].matched is True


def test_public_surface_exports_all_names():
    from pipeline.similarity import SkillMatch as M0
    from pipeline.similarity import SkillMatcher as M
    from pipeline.similarity import SkillMatchReport as R

    assert M is SkillMatcher
    assert R is SkillMatchReport
    assert M0 is SkillMatch
