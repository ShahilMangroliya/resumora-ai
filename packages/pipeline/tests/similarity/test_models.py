import pytest
from pydantic import ValidationError

from pipeline.similarity.models import SkillMatch, SkillMatchReport


def test_skill_match_valid():
    m = SkillMatch(
        jd_skill="python",
        resume_skill="Python",
        similarity=0.92,
        matched=True,
    )
    assert m.jd_skill == "python"
    assert m.matched is True


def test_skill_match_similarity_out_of_range_rejected():
    with pytest.raises(ValidationError):
        SkillMatch(jd_skill="a", resume_skill="b", similarity=1.5, matched=False)


def test_skill_match_negative_similarity_rejected():
    with pytest.raises(ValidationError):
        SkillMatch(jd_skill="a", resume_skill="b", similarity=-0.1, matched=False)


def test_skill_match_resume_skill_may_be_empty_when_resume_has_no_skills():
    # Documented: if the resume has zero skills, "best resume skill" is "".
    m = SkillMatch(jd_skill="python", resume_skill="", similarity=0.0, matched=False)
    assert m.resume_skill == ""


def test_skill_match_report_match_rate():
    report = SkillMatchReport(
        required_matched=[SkillMatch(jd_skill="a", resume_skill="A", similarity=0.9, matched=True)],
        required_missing=[SkillMatch(jd_skill="b", resume_skill="x", similarity=0.1, matched=False)],
        nice_to_have_matched=[],
        nice_to_have_missing=[],
        match_rate=0.5,
    )
    assert report.match_rate == 0.5


def test_skill_match_report_match_rate_bounded():
    with pytest.raises(ValidationError):
        SkillMatchReport(
            required_matched=[],
            required_missing=[],
            nice_to_have_matched=[],
            nice_to_have_missing=[],
            match_rate=1.5,
        )
