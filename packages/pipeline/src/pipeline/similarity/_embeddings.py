from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingBackend(Protocol):
    """Anything that turns a list of texts into an (N, D) array of unit vectors."""

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerBackend:
    """Default backend: wraps `sentence_transformers.SentenceTransformer`.

    Imports are lazy so this module loads instantly in unit tests that inject
    a fake backend.
    """

    def __init__(self, model_name: str, *, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        # normalize_embeddings=True makes cosine reduce to a dot product, which
        # keeps the matcher arithmetic numerically tight.
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True),
            dtype=np.float64,
        )
