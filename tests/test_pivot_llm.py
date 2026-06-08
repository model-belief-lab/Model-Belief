from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

import pytest

from model_belief.config.derive import derive_token_maps
from model_belief.config.models import (
    AlternativeConfig,
    ChoiceMapConfig,
    ChoiceSetConfig,
    DerivedChoiceMap,
    LlmJudgeConfig,
    PivotConfig,
    TokenPolicyConfig,
)
from model_belief.pivot.base import PivotContext, PivotInputError, resolve_pivot
from model_belief.pivot.llm_judge import LlmJudgeFinder


def make_logprobs_content(tokens: Sequence[str]) -> List[Dict[str, Any]]:
    return [{"token": t} for t in tokens]


def make_choice_map_llm_judge() -> DerivedChoiceMap:
    alternatives = [
        AlternativeConfig(id="pampers", label="Pampers", target_tokens=["P", "Pamp", "Pam"]),
        AlternativeConfig(id="huggies", label="Huggies", target_tokens=["H"]),
        AlternativeConfig(id="luvs", label="Luvs", target_tokens=["L"]),
        AlternativeConfig(id="none", label="None", target_tokens=["None", "none", "Neither", "neither"]),
    ]
    token_policy = TokenPolicyConfig(case_sensitive=True, allow_leading_space_variant=True)

    pivot = PivotConfig(
        active="llm_judge",
        llm_judge=LlmJudgeConfig(return_format="json"),
        first_token_available=None,
        custom=None,
    )

    cfg = ChoiceMapConfig(
        version=1,
        name="llm_judge_test",
        description="unit test",
        choice_set=ChoiceSetConfig(id="diapers", alternatives=alternatives),
        token_policy=token_policy,
        pivot=pivot,
    )

    derived_tokens = derive_token_maps(cfg.choice_set.alternatives, cfg.token_policy)
    return DerivedChoiceMap(config=cfg, tokens=derived_tokens)


def test_llm_judge_prefers_ctx_judge_fn():
    choice_map = make_choice_map_llm_judge()
    logprobs = make_logprobs_content(["Intro", "Answer:", " H"])

    def judge_fn(**kwargs) -> int:
        return 2

    finder = LlmJudgeFinder(max_tokens_to_show=50)
    ctx = PivotContext(text="Answer: Huggies", judge_fn=judge_fn, metadata={"anything": "ok"})

    res = resolve_pivot(finder, logprobs, choice_map, ctx)
    assert res.index == 2
    assert res.choice_id == "huggies"
    assert res.method == "llm_judge"
    assert res.debug["used_openai_judge"] is False


def test_llm_judge_fallback_calls_internal_openai_judge(monkeypatch):
    choice_map = make_choice_map_llm_judge()
    logprobs = make_logprobs_content(["Intro", "Answer:", " P"])

    dummy_models_config = SimpleNamespace(
        openai=SimpleNamespace(
            api_key_env="OPENAI_API_KEY",
            pivot_judge=SimpleNamespace(model="dummy-judge-model", temperature=0.0, max_completion_tokens=64),
        )
    )

    import model_belief.pivot.llm_judge as lj_mod

    def fake_openai_judge_pivot_index(**kwargs) -> int:
        return 2

    monkeypatch.setattr(lj_mod, "_openai_judge_pivot_index", fake_openai_judge_pivot_index)

    finder = LlmJudgeFinder(max_tokens_to_show=50)
    ctx = PivotContext(text="Answer: Pampers", judge_fn=None, metadata={"models_config": dummy_models_config})

    res = resolve_pivot(finder, logprobs, choice_map, ctx)
    assert res.index == 2
    assert res.choice_id == "pampers"
    assert res.debug["used_openai_judge"] is True
    assert res.debug["judge_model"] == "dummy-judge-model"


def test_llm_judge_missing_metadata_raises():
    choice_map = make_choice_map_llm_judge()
    logprobs = make_logprobs_content(["Intro", "Answer:", " P"])

    finder = LlmJudgeFinder()
    ctx = PivotContext(text="Answer: Pampers", judge_fn=None, metadata=None)

    with pytest.raises(PivotInputError):
        resolve_pivot(finder, logprobs, choice_map, ctx)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or os.getenv("RUN_OPENAI_INTEGRATION") != "1",
    reason="Integration test requires OPENAI_API_KEY and RUN_OPENAI_INTEGRATION=1",
)
def test_llm_judge_real_openai_call_with_trace(tmp_path):
    """Opt-in integration test that calls OpenAI and prints judge I/O for inspection."""

    models_yaml = """\
    provider: openai
    
    openai:
      api_key_env: OPENAI_API_KEY
    
      response:
        model: gpt-5.2-2025-12-11
        temperature: 1.0
        max_completion_tokens: 16
        logprobs: true
        top_logprobs: 5
        passthrough: {}
    
      pivot_judge:
        model: gpt-5-mini-2025-08-07
        temperature: 0.0
        max_completion_tokens: 64
    """
    models_path = tmp_path / "models.yaml"
    models_path.write_text(models_yaml, encoding="utf-8")

    choice_map = make_choice_map_llm_judge()

    # Complex token stream (>= 12 tokens), but ONLY one token belongs to the token universe: " H"
    logprobs = make_logprobs_content(
        [
            "In",
            "this",
            "example",
            ",",
            "we",
            "consider",
            "Pampers",
            "and",
            "Huggies",
            ".",
            "After",
            "thinking",
            ",",
            "Answer:",
            " H",
            ".",
        ]
    )

    finder = LlmJudgeFinder(max_tokens_to_show=80)
    ctx = PivotContext(
        text=(
            "In this example, we consider Pampers and Huggies. "
            "After thinking, Answer: I choose Huggies."
        ),
        judge_fn=None,
        metadata={"models_yaml_path": str(models_path)},
    )

    try:
        res = resolve_pivot(finder, logprobs, choice_map, ctx)
    finally:
        trace = ctx.metadata.get("_openai_judge_trace") if isinstance(ctx.metadata, dict) else None
        print("\n--- OpenAI judge trace ---")
        if isinstance(trace, dict):
            print("Judge model:", trace.get("judge_model"))
            print("System prompt:\n", trace.get("system_prompt"))
            print("User prompt (JSON):\n", trace.get("user_prompt"))
            print("Raw output text:\n", trace.get("raw_output_text"))
            print("Raw response json:\n", trace.get("raw_response_json"))

    assert res.index == 14
    assert res.token == " H"
    assert res.choice_id == "huggies"
    assert res.choice_id == "huggies"
    assert res.method == "llm_judge"
    assert res.debug.get("used_openai_judge") is True
