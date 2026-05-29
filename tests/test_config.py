import pytest
from pydantic import ValidationError


def test_defaults_applied():
    from dwp.config import Settings

    s = Settings(doubleword_api_key="test-key", _env_file=None)
    assert s.mode == "realtime"
    assert s.backend == "local"
    assert s.doubleword_base_url == "https://api.doubleword.ai/v1"
    assert s.phoenix_collector_endpoint == "http://localhost:6006"
    assert s.max_concurrency == 8


def test_missing_api_key_raises():
    from dwp.config import Settings

    with pytest.raises((ValidationError, Exception)):
        Settings(_env_file=None)


def test_env_var_overrides_mode(monkeypatch):
    monkeypatch.setenv("MODE", "batch")
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "test-key")

    from importlib import reload
    import dwp.config as cfg_module
    reload(cfg_module)

    # Settings picks up the monkeypatched env
    s = cfg_module.Settings()
    assert s.mode == "batch"


def test_model_defaults():
    from dwp.config import Settings

    s = Settings(doubleword_api_key="test-key", _env_file=None)
    assert s.model_chat == "deepseek-ai-deepseek-v4-pro"
    assert s.model_judge == "qwen3-5-7b-instruct"
