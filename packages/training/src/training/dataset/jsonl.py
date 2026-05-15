import json
from collections.abc import Iterable
from pathlib import Path

from training.dataset.schema import Pair


def write_pairs(path: Path, pairs: Iterable[Pair]) -> None:
    """Append pairs to a JSONL file, one record per line.

    Creates the parent directory if missing. Each write is a flush, so
    interrupting the generator leaves a valid partial file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for pair in pairs:
            f.write(pair.model_dump_json())
            f.write("\n")


def read_pairs(path: Path) -> list[Pair]:
    """Load every record from a JSONL file."""
    if not path.exists():
        return []
    pairs: list[Pair] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            pairs.append(Pair.model_validate_json(line))
    return pairs


def read_existing_ids(path: Path) -> set[str]:
    """Return the set of pair_ids already present on disk (empty if missing)."""
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.add(json.loads(line)["pair_id"])
    return ids
