from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from pipeline.extraction.client import OllamaClient

from training.dataset.jsonl import read_existing_ids, write_pairs
from training.dataset.roles import ROLES, RoleSpec
from training.dataset.schema import Label
from training.dataset.synthesize import generate_pair

_LABELS: tuple[Label, ...] = ("strong", "partial", "weak")


@dataclass(frozen=True)
class _Target:
    role: RoleSpec
    seniority: str
    label: Label
    seed: int


def iter_targets(seed_start: int = 0) -> Iterator[_Target]:
    """Walk (role × seniority × label) on an infinite loop, incrementing `seed` each yield.

    Phase 2's 15-role catalog × seniorities × 3 labels = 150 unique triples in one pass.
    For targets above 150, the iterator wraps and re-yields each combo with a new seed —
    `generate_pair` is deterministic on (role, seniority, label, seed), so the resulting
    pair_ids stay unique. The caller is responsible for stopping (e.g. via `written >= needed`).
    """
    seed = seed_start
    while True:
        for role in ROLES:
            for seniority in role.seniorities:
                for label in _LABELS:
                    yield _Target(role=role, seniority=seniority, label=label, seed=seed)
                    seed += 1


def _default_client() -> OllamaClient:
    return OllamaClient()


def run_generate(out: Path, target: int) -> int:
    """Generate up to `target` pairs to `out`, skipping ones already present.

    Returns the total number of records on disk after the run.
    """
    existing = read_existing_ids(out)
    needed = max(target - len(existing), 0)
    if needed == 0:
        return len(existing)

    client = _default_client()
    written = 0
    batch = []
    for target_spec in iter_targets():
        if written >= needed:
            break
        pair = generate_pair(
            target_spec.role,
            target_spec.seniority,
            target_spec.label,
            client=client,
            seed=target_spec.seed,
        )
        if pair.pair_id in existing:
            continue
        batch.append(pair)
        existing.add(pair.pair_id)
        written += 1
        if len(batch) >= 10:
            write_pairs(out, batch)
            batch = []

    if batch:
        write_pairs(out, batch)
    return len(existing)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="training.dataset.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("generate", help="generate synthetic pairs")
    gen.add_argument("--out", type=Path, required=True)
    gen.add_argument("--target", type=int, required=True)
    args = parser.parse_args(argv)

    if args.cmd == "generate":
        total = run_generate(args.out, args.target)
        print(f"records on disk: {total}")


if __name__ == "__main__":
    main()
