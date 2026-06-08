from __future__ import annotations

from dataclasses import dataclass
from platform import system
from typing import Any, Dict, List, Optional, Sequence

import os
import pytest

from model_belief.llm import LlmConfigError, LlmProviderError
from model_belief.llm import openai_generate_with_logprobs


# ---- Dummy config object matching your models.yaml shape ----

class DummyModelsConfig:
    provider = "openai"

    class openai:
        api_key_env = "OPENAI_API_KEY"

        class response:
            model = "dummy-model"
            temperature = 1.0
            max_completion_tokens = 16
            logprobs = True
            top_logprobs = 5
            passthrough = {"reasoning": {"effort": "none"}}


# ---- Fake OpenAI Response object ----

@dataclass
class FakeContent:
    type: str = "output_text"
    text: str = "Answer: H"
    logprobs: Any = None


@dataclass
class FakeOutputItem:
    content: List[FakeContent]


class FakeResp:
    # exercise both .output_text and output scanning
    output_text: str = ""

    def __init__(self, *, text: str, logprobs_obj: Any):
        self.output = [FakeOutputItem(content=[FakeContent(type="output_text", text=text, logprobs=logprobs_obj)])]


class FakeResponses:
    def __init__(self):
        self.last_kwargs: Optional[Dict[str, Any]] = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        # Return tokens/logprobs as a simple list of dicts compatible with your pivot tests
        fake_logprobs = [{"token": " In"}, {"token": " H"}]
        return FakeResp(text="In ... Answer: H", logprobs_obj=fake_logprobs)


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


def _patch_get_openai_client(monkeypatch):
    import model_belief.llm.openai_response as mod

    fake_client = FakeClient()

    def fake_get_openai_client(*, models_yaml_path=None, models_config=None):
        assert models_config is not None
        return fake_client, models_config

    monkeypatch.setattr(mod, "get_openai_client", fake_get_openai_client)
    return fake_client


def test_openai_generate_with_logprobs_builds_request(monkeypatch):
    fake_client = _patch_get_openai_client(monkeypatch)

    res = openai_generate_with_logprobs(
        system_prompt="sys",
        user_prompt="usr",
        models_config=DummyModelsConfig(),
    )

    assert res.text.startswith("In")
    assert len(res.logprobs_content) == 2

    # Validate request keys
    req = fake_client.responses.last_kwargs
    assert req is not None
    assert req["model"] == "dummy-model"
    assert req["logprobs"] is True
    assert req["top_logprobs"] == 5
    assert req["max_output_tokens"] == 16

    # From config passthrough
    assert req["reasoning"] == {"effort": "none"}

    # Temperature included by default (unless overridden)
    assert req["temperature"] == 1.0


def test_openai_generate_passthrough_and_overrides(monkeypatch):
    fake_client = _patch_get_openai_client(monkeypatch)

    _ = openai_generate_with_logprobs(
        system_prompt="sys",
        user_prompt="usr",
        models_config=DummyModelsConfig(),
        passthrough={"foo": "bar", "temperature": 0.2},
        temperature=0.7,  # override should win over passthrough and config
    )

    req = fake_client.responses.last_kwargs
    assert req is not None
    assert req["foo"] == "bar"
    assert req["temperature"] == 0.7  # override wins


def test_openai_generate_requires_logprobs_true(monkeypatch):
    fake_client = _patch_get_openai_client(monkeypatch)

    class BadCfg(DummyModelsConfig):
        class openai(DummyModelsConfig.openai):
            class response(DummyModelsConfig.openai.response):
                logprobs = False

    with pytest.raises(LlmConfigError):
        openai_generate_with_logprobs(
            system_prompt="sys",
            user_prompt="usr",
            models_config=BadCfg(),
        )


def test_openai_generate_requires_positive_top_logprobs(monkeypatch):
    fake_client = _patch_get_openai_client(monkeypatch)

    class BadCfg(DummyModelsConfig):
        class openai(DummyModelsConfig.openai):
            class response(DummyModelsConfig.openai.response):
                top_logprobs = 0

    with pytest.raises(LlmConfigError):
        openai_generate_with_logprobs(
            system_prompt="sys",
            user_prompt="usr",
            models_config=BadCfg(),
        )


def test_openai_generate_provider_error(monkeypatch):
    import model_belief.llm.openai_response as mod

    def fake_get_openai_client(*, models_yaml_path=None, models_config=None):
        class BadClient:
            class responses:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("boom")

        return BadClient(), models_config

    monkeypatch.setattr(mod, "get_openai_client", fake_get_openai_client)

    with pytest.raises(LlmProviderError):
        openai_generate_with_logprobs(
            system_prompt="sys",
            user_prompt="usr",
            models_config=DummyModelsConfig(),
        )



system_prompt = ("You are making a routine purchase decision in a store. "
                 "You are choosing among products based on overall impressions such as quality, comfort, and price."
                 "Your preferences are not perfectly sharp: when options are similar, small differences may not lead to a clear or consistent choice."
                 "There is no single right answer. Indicate which option you would choose, if any. "
                 "Answer naturally, as you would in a quick shopping situation. Do not choose more than one option.")

user_prompt = ("You see the following baby diaper brands in the store: Pampers and Huggies. "
               "Pampers diapers are generally described as soft and include a wetness indicator (though experiences vary), and are often perceived as slightly more trusted and premium overall. "
               "Huggies diapers are often noted for having a snug fit and for helping to prevent leaks, though the fit and effectiveness can depend on the baby. "
               "When choosing diapers, you weigh functionality, brand trust, and saving money. Comfort and reliability often matter more to you than small price differences, but as price gaps grow, saving money becomes increasingly important. "
               "There is no clear cutoff for when a price difference feels “small” versus “large,” and it can depend on your overall impression in the moment. "
               "The unit prices are 35.6 cents per Pampers diaper and 30.0 cents per Huggies diaper. "
               "Indicate which brand you would choose to buy, or choose “neither” if you prefer other diaper brands. Answer naturally.")

@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_OPENAI_INTEGRATION") != "1" or not os.getenv("OPENAI_API_KEY"),
    reason="Requires OPENAI_API_KEY and RUN_OPENAI_INTEGRATION=1",
)
def test_openai_response_integration_returns_text_and_logprobs():
    # res = openai_generate_with_logprobs(
    #     system_prompt=system_prompt,
    #     user_prompt=user_prompt,
    #     models_yaml_path="configs/models.yaml",
    #     passthrough={
    #         # you wanted reasoning effort none
    #         "reasoning": {"effort": "none"},
    #         # do NOT set temperature here (some models reject it)
    #     },
    # )
    res = openai_generate_with_logprobs(
        input="Choose one letter: H or P",
        instructions="Return exactly one letter.",
        models_yaml_path="configs/models.yaml",
        passthrough={
            # you wanted reasoning effort none
            "reasoning": {"effort": "none"},
            # do NOT set temperature here (some models reject it)
        },
    )


    print("\n--- OpenAI trace ---")
    print("\n--- Requested parameters ---")
    print(res.debug.get("request_parameters"))
    print("\n--- Generated Text ---")
    print(res.text)
    print("\n--- OpenAI logprobs ---")
    print(res.logprobs_content)
    print("\n--- Full debug info ---")
    print(res.debug)

    assert isinstance(res.text, str) and res.text.strip() != ""
    assert res.logprobs_content is not None
    assert hasattr(res.logprobs_content, "__iter__")
    assert res.debug.get("provider") == "openai"
