import pytest

from api.config import Settings, load_settings


def test_load_settings_defaults(monkeypatch: pytest.MonkeyPatch):
    for var in (
        "RESUMEFIT_SCORER_REPO",
        "RESUMEFIT_SCORER_DEVICE",
        "RESUMEFIT_MATCHER_DEVICE",
        "RESUMEFIT_WARMUP_ON_STARTUP",
        "RESUMEFIT_OLLAMA_URL",
        "RESUMEFIT_OLLAMA_MODEL",
        "RESUMEFIT_OLLAMA_TIMEOUT",
        "RESUMEFIT_CORS_ORIGINS",
    ):
        monkeypatch.delenv(var, raising=False)
    settings = load_settings()
    assert isinstance(settings, Settings)
    assert settings.scorer_repo == "distilbert-base-uncased"
    assert settings.scorer_device == "cpu"
    assert settings.matcher_device == "cpu"
    assert settings.warmup_on_startup is False
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_model == "llama3.2:3b"
    assert settings.ollama_timeout == 30.0
    assert settings.cors_origins == ("http://localhost:3000",)


def test_load_settings_reads_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RESUMEFIT_SCORER_REPO", "user/resumefit-distilbert-lora")
    monkeypatch.setenv("RESUMEFIT_SCORER_DEVICE", "cuda")
    monkeypatch.setenv("RESUMEFIT_MATCHER_DEVICE", "cuda")
    monkeypatch.setenv("RESUMEFIT_WARMUP_ON_STARTUP", "true")
    monkeypatch.setenv("RESUMEFIT_OLLAMA_URL", "http://ollama:11434")
    monkeypatch.setenv("RESUMEFIT_OLLAMA_MODEL", "qwen2.5:7b")
    monkeypatch.setenv("RESUMEFIT_OLLAMA_TIMEOUT", "15.5")
    monkeypatch.setenv(
        "RESUMEFIT_CORS_ORIGINS",
        "https://resumefit.vercel.app, http://localhost:3000",
    )
    settings = load_settings()
    assert settings.scorer_repo == "user/resumefit-distilbert-lora"
    assert settings.scorer_device == "cuda"
    assert settings.matcher_device == "cuda"
    assert settings.warmup_on_startup is True
    assert settings.ollama_base_url == "http://ollama:11434"
    assert settings.ollama_model == "qwen2.5:7b"
    assert settings.ollama_timeout == 15.5
    assert settings.cors_origins == (
        "https://resumefit.vercel.app",
        "http://localhost:3000",
    )


@pytest.mark.parametrize("raw,expected", [
    ("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True),
    ("0", False), ("false", False), ("no", False), ("", False), ("anything", False),
])
def test_warmup_flag_parses(monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool):
    monkeypatch.setenv("RESUMEFIT_WARMUP_ON_STARTUP", raw)
    assert load_settings().warmup_on_startup is expected


def test_cors_origins_blanks_collapse(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RESUMEFIT_CORS_ORIGINS", " , https://a.com ,, https://b.com , ")
    assert load_settings().cors_origins == ("https://a.com", "https://b.com")
