from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from training.dataset.schema import Pair


@dataclass
class Report:
    """Summary of a dataset's shape — used as a gate before publishing."""

    total: int
    label_counts: dict[str, int]
    unique_roles: int
    duplicate_pair_ids: int
    duplicate_resume_texts: int
    duplicate_jd_texts: int


def validate(pairs: Iterable[Pair]) -> Report:
    pairs = list(pairs)
    label_counts = dict(Counter(p.label for p in pairs))
    roles = {p.role for p in pairs}

    id_counter = Counter(p.pair_id for p in pairs)
    dup_ids = sum(c - 1 for c in id_counter.values() if c > 1)

    resume_counter = Counter(p.resume_text for p in pairs)
    dup_resumes = sum(c - 1 for c in resume_counter.values() if c > 1)

    jd_counter = Counter(p.jd_text for p in pairs)
    dup_jds = sum(c - 1 for c in jd_counter.values() if c > 1)

    return Report(
        total=len(pairs),
        label_counts=label_counts,
        unique_roles=len(roles),
        duplicate_pair_ids=dup_ids,
        duplicate_resume_texts=dup_resumes,
        duplicate_jd_texts=dup_jds,
    )
