from __future__ import annotations

import os
import pytest

from model_belief.llm import LlmConfigError
from model_belief.llm import generate_with_logprobs


class DummyCfg:
    provider = "openai"


def test_generate_dispatches_openai(monkeypatch):
    import model_belief.llm.generate as gen_mod

    # Patch load_models
    monkeypatch.setattr(gen_mod, "load_models", lambda path: DummyCfg())

    # Patch openai_generate_with_logprobs
    def fake_openai_generate_with_logprobs(**kwargs):
        return "OK"

    monkeypatch.setattr(gen_mod, "openai_generate_with_logprobs", fake_openai_generate_with_logprobs)

    out = generate_with_logprobs(
        system_prompt="sys",
        user_prompt="usr",
        models_yaml_path="dummy.yaml",
    )
    assert out == "OK"


def test_generate_missing_provider(monkeypatch):
    import model_belief.llm.generate as gen_mod

    class NoProviderCfg:
        provider = None

    monkeypatch.setattr(gen_mod, "load_models", lambda path: NoProviderCfg())

    with pytest.raises(LlmConfigError):
        generate_with_logprobs(
            system_prompt="sys",
            user_prompt="usr",
            models_yaml_path="dummy.yaml",
        )


def test_generate_unsupported_provider(monkeypatch):
    import model_belief.llm.generate as gen_mod

    class BadCfg:
        provider = "something_else"

    monkeypatch.setattr(gen_mod, "load_models", lambda path: BadCfg())

    with pytest.raises(LlmConfigError):
        generate_with_logprobs(
            system_prompt="sys",
            user_prompt="usr",
            models_yaml_path="dummy.yaml",
        )



@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_OPENAI_INTEGRATION") != "1" or not os.getenv("OPENAI_API_KEY"),
    reason="Requires OPENAI_API_KEY and RUN_OPENAI_INTEGRATION=1",
)
def test_generate_facade_integration_dispatches_and_runs():
    res = generate_with_logprobs(
        system_prompt="You are a strict assistant.",
        user_prompt="Reply with exactly one character from {H, P}: P",
        models_yaml_path="configs/models.yaml",
        passthrough={"reasoning": {"effort": "none"}},
    )

    assert isinstance(res.text, str) and res.text.strip() != ""
    assert res.debug.get("provider") == "openai"