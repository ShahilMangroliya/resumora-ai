from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Runtime configuration loaded from RESUMEFIT_* environment variables."""

    scorer_repo: str = "distilbert-base-uncased"
    scorer_device: str = "cpu"
    matcher_device: str = "cpu"
    warmup_on_startup: bool = False
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout: float = 30.0
    cors_origins: tuple[str, ...] = ("http://localhost:3000",)


_TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_VALUES


def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    parts = tuple(p.strip() for p in raw.split(",") if p.strip())
    return parts or default


def load_settings() -> Settings:
    """Read RESUMEFIT_* env vars and return a frozen Settings.

    Loaded fresh on every call — caching belongs to `dependencies.py` so tests
    can monkeypatch env vars between requests.
    """
    return Settings(
        scorer_repo=os.getenv("RESUMEFIT_SCORER_REPO", "distilbert-base-uncased"),
        scorer_device=os.getenv("RESUMEFIT_SCORER_DEVICE", "cpu"),
        matcher_device=os.getenv("RESUMEFIT_MATCHER_DEVICE", "cpu"),
        warmup_on_startup=_bool_env("RESUMEFIT_WARMUP_ON_STARTUP", False),
        ollama_base_url=os.getenv("RESUMEFIT_OLLAMA_URL", "http://localhost:11434"),
        ollama_model=os.getenv("RESUMEFIT_OLLAMA_MODEL", "llama3.2:3b"),
        ollama_timeout=float(os.getenv("RESUMEFIT_OLLAMA_TIMEOUT", "30.0")),
        cors_origins=_csv_env("RESUMEFIT_CORS_ORIGINS", ("http://localhost:3000",)),
    )
