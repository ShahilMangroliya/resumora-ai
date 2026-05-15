from pathlib import Path

from training.dataset.jsonl import read_pairs
from training.dataset.schema import Pair

_REPO_ROOT = Path(__file__).resolve().parents[5]
SEED_PATH = _REPO_ROOT / "data" / "gold" / "seed.jsonl"


def load_seed() -> list[Pair]:
    """Load the committed gold seed pairs from data/gold/seed.jsonl."""
    return read_pairs(SEED_PATH)
