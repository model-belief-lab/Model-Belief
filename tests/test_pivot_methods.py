# tests/test_pivot_methods.py
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import pytest

from model_belief.config.models import (
    AlternativeConfig,
    ChoiceMapConfig,
    ChoiceSetConfig,
    CustomPivotConfig,
    DerivedChoiceMap,
    FirstTokenAvailableConfig,
    LlmJudgeConfig,
    PivotConfig,
    TokenPolicyConfig,
)
from model_belief.config.derive import derive_token_maps

from model_belief.pivot.base import PivotContext, PivotValidationError, resolve_pivot
from model_belief.pivot.first_available import FirstTokenAvailableFinder
from model_belief.pivot.llm_judge import LlmJudgeFinder
from model_belief.pivot.custom import CustomFinder


# -----------------------------
# Helpers to build test fixtures
# -----------------------------

def make_choice_map(active: str) -> DerivedChoiceMap:
    """
    Build a minimal DerivedChoiceMap consistent with your diapers example.
    Overlaps are avoided so token->alt is unique, matching V1 constraints.
    """
    alternatives = [
        AlternativeConfig(id="pampers", label="Pampers", target_tokens=["P", "Pamp", "Pam"]),
        AlternativeConfig(id="huggies", label="Huggies", target_tokens=["H"]),
        AlternativeConfig(id="luvs", label="Luvs", target_tokens=["L"]),
        AlternativeConfig(id="none", label="None", target_tokens=["None", "none", "Neither", "neither"]),
    ]
    token_policy = TokenPolicyConfig(case_sensitive=True, allow_leading_space_variant=True)

    pivot = PivotConfig(
        active=active,  # type: ignore[arg-type]
        first_token_available=FirstTokenAvailableConfig(
            anchors=["Answer:", "Final answer:"],
            anchor_policy="last",
        )
        if active == "first_token_available"
        else None,
        llm_judge=LlmJudgeConfig(return_format="json") if active == "llm_judge" else None,
        custom=CustomPivotConfig(function=f"{__name__}:custom_find_pivot") if active == "custom" else None,
    )

    cfg = ChoiceMapConfig(
        version=1,
        name="diapers_test",
        description="unit test",
        choice_set=ChoiceSetConfig(id="diapers", alternatives=alternatives),
        token_policy=token_policy,
        pivot=pivot,
    )

    derived_tokens = derive_token_maps(cfg.choice_set.alternatives, cfg.token_policy)
    return DerivedChoiceMap(config=cfg, tokens=derived_tokens)


def make_logprobs_content(tokens: Sequence[str]) -> List[Dict[str, Any]]:
    """
    Minimal logprobs.content-like sequence.
    Only "token" is required by pivot module.
    """
    return [{"token": t} for t in tokens]


# -----------------------------
# Custom pivot function (Strategy C)
# -----------------------------
def custom_find_pivot(*, text: str | None, logprobs_content: Sequence[Any], context: Dict[str, Any]) -> int:
    """
    Example user-defined pivot finder.
    Here we locate the first universe token after a synthetic marker token '<<ANS>>'.
    """
    # Find marker
    marker_idx = None
    for i, item in enumerate(logprobs_content):
        if isinstance(item, dict) and item.get("token") == "<<ANS>>":
            marker_idx = i
            break
    if marker_idx is None:
        raise ValueError("Marker token <<ANS>> not found")

    token_universe = context["token_universe"]
    case_sensitive = context["token_policy"]["case_sensitive"]

    def norm(t: str) -> str:
        return t if case_sensitive else t.lower()

    for j in range(marker_idx + 1, len(logprobs_content)):
        tok = logprobs_content[j]["token"]
        if norm(tok) in token_universe:
            return j

    raise ValueError("No universe token after marker")


# -----------------------------
# Tests
# -----------------------------

def test_first_token_available_finds_first_universe_token():
    choice_map = make_choice_map(active="first_token_available")

    # Token stream includes a non-universe prelude, then a universe token " P" (leading space variant)
    logprobs = make_logprobs_content(["Hello", "world", "Answer:", " P", "ampers"])

    finder = FirstTokenAvailableFinder(
        anchors=choice_map.config.pivot.first_token_available.anchors,  # type: ignore[union-attr]
        anchor_policy=choice_map.config.pivot.first_token_available.anchor_policy,  # type: ignore[union-attr]
    )

    # Anchor must exist in ctx.text when anchors configured
    ctx = PivotContext(text="Some preface.\nAnswer: Pampers")

    res = resolve_pivot(finder, logprobs, choice_map, ctx)
    assert res.index == 3
    assert res.choice_id == "pampers"
    assert res.token == " P"
    assert res.method == "first_token_available"


def test_llm_judge_uses_injected_judge_fn():
    choice_map = make_choice_map(active="llm_judge")

    logprobs = make_logprobs_content(["Intro", "<<ANS>>", " H", "uggies"])

    def judge_fn(**kwargs) -> int:
        # In a real judge, you'd inspect kwargs["text"], kwargs["tokens"], etc.
        # Here we hardcode the pivot index to the token " H".
        return 2

    finder = LlmJudgeFinder(max_tokens_to_show=100)
    ctx = PivotContext(text="Answer: Huggies", judge_fn=judge_fn)

    res = resolve_pivot(finder, logprobs, choice_map, ctx)
    assert res.index == 2
    assert res.choice_id == "huggies"
    assert res.token == " H"
    assert res.method == "llm_judge"


def test_custom_finder_loads_dotted_callable_and_returns_valid_pivot():
    choice_map = make_choice_map(active="custom")

    # Marker then token " L" should map to luvs
    logprobs = make_logprobs_content(["blah", "<<ANS>>", " L", "uvs"])

    finder = CustomFinder(function_path=f"{__name__}:custom_find_pivot")
    ctx = PivotContext(text="Answer: Luvs")

    res = resolve_pivot(finder, logprobs, choice_map, ctx)
    assert res.index == 2
    assert res.choice_id == "luvs"
    assert res.token == " L"
    assert res.method == "custom"


def test_validation_rejects_non_universe_token():
    choice_map = make_choice_map(active="first_token_available")

    # No universe tokens present at all
    logprobs = make_logprobs_content(["Hello", "world", "Answer:", "XYZ"])

    finder = FirstTokenAvailableFinder(
        anchors=choice_map.config.pivot.first_token_available.anchors,  # type: ignore[union-attr]
        anchor_policy=choice_map.config.pivot.first_token_available.anchor_policy,  # type: ignore[union-attr]
    )
    ctx = PivotContext(text="Answer: Pampers")

    with pytest.raises(Exception):
        resolve_pivot(finder, logprobs, choice_map, ctx)


def test_llm_judge_invalid_index_raises():
    choice_map = make_choice_map(active="llm_judge")
    logprobs = make_logprobs_content(["Answer:", " P"])

    def bad_judge_fn(**kwargs) -> int:
        return 999  # out of range

    finder = LlmJudgeFinder()
    ctx = PivotContext(text="Answer: Pampers", judge_fn=bad_judge_fn)

    with pytest.raises(PivotValidationError):
        resolve_pivot(finder, logprobs, choice_map, ctx)