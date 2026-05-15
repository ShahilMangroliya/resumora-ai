from training.dataset.schema import Pair
from training.dataset.validate import Report, validate


def _pair(pair_id: str, label="strong", role="backend-dev", resume="r", jd="j") -> Pair:
    return Pair(
        pair_id=pair_id,
        resume_text=resume,
        jd_text=jd,
        label=label,
        score=85 if label == "strong" else 55 if label == "partial" else 20,
        role=role,
        seniority="mid",
        source="synthetic",
        generator_model="llama3.2:3b",
        generated_at="2026-05-15T10:00:00Z",
        prompt_seed=0,
    )


def test_validate_returns_a_report_object():
    report = validate([_pair("a")])
    assert isinstance(report, Report)


def test_label_counts_are_correct():
    pairs = [_pair("a"), _pair("b"), _pair("c", label="weak"), _pair("d", label="partial")]
    report = validate(pairs)
    assert report.label_counts == {"strong": 2, "partial": 1, "weak": 1}


def test_role_coverage_counts_unique_roles():
    pairs = [_pair("a"), _pair("b", role="frontend-dev")]
    report = validate(pairs)
    assert report.unique_roles == 2


def test_duplicate_pair_ids_are_flagged():
    pairs = [_pair("a"), _pair("a")]
    report = validate(pairs)
    assert report.duplicate_pair_ids == 1


def test_resumes_repeating_verbatim_are_flagged():
    pairs = [_pair("a", resume="same"), _pair("b", resume="same")]
    report = validate(pairs)
    assert report.duplicate_resume_texts >= 1


def test_total_count_is_correct():
    pairs = [_pair("a"), _pair("b")]
    assert validate(pairs).total == 2
