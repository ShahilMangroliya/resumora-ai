from pathlib import Path

from training.dataset.jsonl import read_existing_ids, read_pairs, write_pairs
from training.dataset.schema import Pair


def _pair(pair_id: str) -> Pair:
    return Pair(
        pair_id=pair_id,
        resume_text="r",
        jd_text="j",
        label="strong",
        score=85,
        role="backend-dev",
        seniority="mid",
        source="synthetic",
        generator_model="llama3.2:3b",
        generated_at="2026-05-15T10:00:00Z",
        prompt_seed=0,
    )


def test_write_then_read_round_trips(tmp_path: Path):
    out = tmp_path / "pairs.jsonl"
    pairs = [_pair("a"), _pair("b")]
    write_pairs(out, pairs)
    assert [p.pair_id for p in read_pairs(out)] == ["a", "b"]


def test_write_appends_to_existing_file(tmp_path: Path):
    out = tmp_path / "pairs.jsonl"
    write_pairs(out, [_pair("a")])
    write_pairs(out, [_pair("b")])
    assert [p.pair_id for p in read_pairs(out)] == ["a", "b"]


def test_read_existing_ids_returns_set(tmp_path: Path):
    out = tmp_path / "pairs.jsonl"
    write_pairs(out, [_pair("a"), _pair("b")])
    assert read_existing_ids(out) == {"a", "b"}


def test_read_existing_ids_returns_empty_set_when_missing(tmp_path: Path):
    assert read_existing_ids(tmp_path / "nope.jsonl") == set()
