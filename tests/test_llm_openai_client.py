from __future__ import annotations

import os
import types
import pytest

from model_belief.llm import LlmConfigError, LlmProviderError
from model_belief.llm import get_openai_client


class DummyModelsConfig:
    provider = "openai"

    class openai:
        api_key_env = "OPENAI_API_KEY_TEST"


def test_get_openai_client_missing_config_args():
    with pytest.raises(LlmConfigError):
        get_openai_client(models_yaml_path=None, models_config=None)


def test_get_openai_client_missing_openai_section():
    class BadCfg:
        provider = "openai"

    with pytest.raises(LlmConfigError):
        get_openai_client(models_config=BadCfg())


def test_get_openai_client_missing_api_key_env():
    class BadCfg:
        provider = "openai"

        class openai:
            api_key_env = ""

    with pytest.raises(LlmConfigError):
        get_openai_client(models_config=BadCfg())


def test_get_openai_client_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY_TEST", raising=False)
    with pytest.raises(LlmConfigError):
        get_openai_client(models_config=DummyModelsConfig())


def test_get_openai_client_sdk_missing(monkeypatch):
    # Provide key so we get past env check
    monkeypatch.setenv("OPENAI_API_KEY_TEST", "dummy")

    # Make importing openai fail
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("no openai")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(LlmProviderError):
        get_openai_client(models_config=DummyModelsConfig())



@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_OPENAI_INTEGRATION") != "1" or not os.getenv("OPENAI_API_KEY"),
    reason="Requires OPENAI_API_KEY and RUN_OPENAI_INTEGRATION=1",
)
def test_openai_client_integration_can_create_client():
    # Use your repo config path (adjust if your file lives elsewhere)
    client, cfg = get_openai_client(models_yaml_path="configs/models.yaml")
    assert client is not None
    assert hasattr(client, "responses")