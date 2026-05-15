import itertools
from pathlib import Path

from training.dataset import cli
from training.dataset.jsonl import read_pairs


def _expected_first_pass_combos():
    from training.dataset.roles import ROLES

    return {
        (role.slug, sen, label)
        for role in ROLES
        for sen in role.seniorities
        for label in ("strong", "partial", "weak")
    }


def test_iter_targets_first_pass_covers_every_role_seniority_and_label():
    expected = _expected_first_pass_combos()
    first_pass = list(itertools.islice(cli.iter_targets(seed_start=0), len(expected)))
    seen = {(t.role.slug, t.seniority, t.label) for t in first_pass}
    assert seen == expected


def test_iter_targets_is_infinite_so_targets_above_first_pass_are_reachable():
    expected = _expected_first_pass_combos()
    # Pull one more than the first-pass size — this would StopIteration if finite.
    extra = next(itertools.islice(cli.iter_targets(seed_start=0), len(expected), len(expected) + 1))
    assert extra is not None
    assert extra.seed == len(expected)


def test_run_generate_writes_target_count_then_stops(tmp_path: Path, monkeypatch, fake_ollama):
    payloads = [{"resume_text": f"r{i}", "jd_text": f"j{i}"} for i in range(100)]
    client = fake_ollama(payloads)
    monkeypatch.setattr(cli, "_default_client", lambda: client)

    out = tmp_path / "pairs.jsonl"
    written = cli.run_generate(out=out, target=5)

    assert written == 5
    on_disk = read_pairs(out)
    assert len(on_disk) == 5
    labels = {p.label for p in on_disk}
    assert labels <= {"strong", "partial", "weak"}


def test_run_generate_resumes_and_does_not_duplicate(tmp_path: Path, monkeypatch, fake_ollama):
    payloads = [{"resume_text": f"r{i}", "jd_text": f"j{i}"} for i in range(100)]
    client = fake_ollama(payloads)
    monkeypatch.setattr(cli, "_default_client", lambda: client)

    out = tmp_path / "pairs.jsonl"
    cli.run_generate(out=out, target=3)
    cli.run_generate(out=out, target=6)

    on_disk = read_pairs(out)
    assert len(on_disk) == 6
    ids = [p.pair_id for p in on_disk]
    assert len(ids) == len(set(ids))
