import numpy as np
import pytest

from pipeline.similarity._embeddings import EmbeddingBackend, SentenceTransformerBackend


class _FakeBackend:
    """Maps each unique text to a fixed orthogonal-ish unit vector."""

    def __init__(self) -> None:
        self._seen: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for t in texts:
            if t not in self._seen:
                vec = np.zeros(8)
                vec[hash(t) % 8] = 1.0
                self._seen[t] = vec
            rows.append(self._seen[t])
        return np.stack(rows)


def test_fake_backend_returns_unit_vectors():
    backend: EmbeddingBackend = _FakeBackend()
    out = backend.encode(["python", "java"])
    assert out.shape == (2, 8)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0])


@pytest.mark.integration
def test_sentence_transformer_backend_real_model():
    """Loads sentence-transformers/all-MiniLM-L6-v2. Gated; skipped by default."""
    backend = SentenceTransformerBackend("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    out = backend.encode(["python", "Python"])
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0], atol=1e-3)
    sim = float(out[0] @ out[1])
    assert sim > 0.9
