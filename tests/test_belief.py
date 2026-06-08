# tests/test_belief.py
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import pytest

from model_belief.config.models import (
    AlternativeConfig,
    ChoiceMapConfig,
    ChoiceSetConfig,
    DerivedChoiceMap,
    PivotConfig,
    TokenPolicyConfig,
)
from model_belief.config.derive import derive_token_maps
from model_belief.belief.belief import compute_belief
from model_belief.belief.errors import BeliefInputError, BeliefParseError


def make_choice_map_diapers() -> DerivedChoiceMap:
    """
    Choice universe includes both with-leading-space and without-leading-space variants
    because TokenPolicy allows leading-space variants.
    """
    alternatives = [
        AlternativeConfig(id="pampers", label="Pampers", target_tokens=["P", "Pamp", "Pam"]),
        AlternativeConfig(id="huggies", label="Huggies", target_tokens=["H"]),
        AlternativeConfig(id="luvs", label="Luvs", target_tokens=["L"]),
        AlternativeConfig(id="none", label="None", target_tokens=["None", "none", "Neither", "neither"]),
    ]
    token_policy = TokenPolicyConfig(case_sensitive=True, allow_leading_space_variant=True)

    cfg = ChoiceMapConfig(
        version=1,
        name="diapers_test",
        description="unit test",
        choice_set=ChoiceSetConfig(id="diapers", alternatives=alternatives),
        token_policy=token_policy,
        pivot=PivotConfig(active="first_token_available", first_token_available=None, llm_judge=None, custom=None),
    )

    derived_tokens = derive_token_maps(cfg.choice_set.alternatives, cfg.token_policy)
    return DerivedChoiceMap(config=cfg, tokens=derived_tokens)


def make_logprobs_content_with_pivot(
    pivot_top_logprobs: Sequence[Dict[str, Any]],
    length: int = 5,
    pivot_index: int = 2,
) -> List[Dict[str, Any]]:
    """
    Minimal provider-native logprobs_content structure expected by compute_belief:
      logprobs_content[pivot_index]["top_logprobs"] = [{token, logprob}, ...]
    Other positions can be empty dicts.
    """
    content: List[Dict[str, Any]] = [{"token": f"t{i}"} for i in range(length)]
    content[pivot_index] = {"token": "PIVOT", "top_logprobs": list(pivot_top_logprobs)}
    return content


def softmax_probs(logits: List[float]) -> List[float]:
    m = max(logits)
    ws = [math.exp(z - m) for z in logits]
    Z = sum(ws)
    return [w / Z for w in ws]


def test_compute_belief_softmax_over_choice_universe_and_aggregate_by_choice():
    """
    - Only tokens in choice_universe AND mapped to a choice participate in softmax.
    - Probabilities are normalized to sum to 1 over choices.
    - Multiple tokens mapping to the same choice are aggregated.
    """
    choice_map = make_choice_map_diapers()

    # Candidate tokens at pivot:
    # - " P" and " Pamp" both map to pampers
    # - " H" maps to huggies
    # - "," is outside universe and should be ignored (not in softmax)
    pivot_top = [
        {"token": " P", "logprob": 2.0},      # logit
        {"token": " Pamp", "logprob": 1.0},   # logit (same choice as pampers)
        {"token": " H", "logprob": 0.0},      # logit
        {"token": ",", "logprob": 10.0},      # outside universe; ignored even though huge
    ]
    logprobs_content = make_logprobs_content_with_pivot(pivot_top, pivot_index=2)

    belief = compute_belief(logprobs_content=logprobs_content, pivot_index=2, choice_map=choice_map)

    # Softmax should be computed ONLY over in-universe mapped tokens: [" P", " Pamp", " H"]
    # logits = [2.0, 1.0, 0.0]
    pP, pPamp, pH = softmax_probs([2.0, 1.0, 0.0])

    # Pampers aggregates " P" + " Pamp"
    expected_pampers = pP + pPamp
    expected_huggies = pH

    got = {it.alt_id: it.prob for it in belief.items}

    assert pytest.approx(sum(got.values()), rel=1e-9, abs=1e-12) == 1.0
    assert pytest.approx(got["pampers"], rel=1e-9, abs=1e-12) == expected_pampers
    assert pytest.approx(got["huggies"], rel=1e-9, abs=1e-12) == expected_huggies

    # Ensure outside-universe token did not affect distribution
    assert belief.debug["normalization"] == "softmax_over_choice_universe"
    assert belief.debug["num_candidates_in_universe"] == 3
    assert belief.debug["num_candidates_outside_universe"] == 1

    # By construction now
    assert belief.total_mass == 1.0
    assert belief.missing_mass == 0.0


def test_compute_belief_raises_when_no_valid_tokens_in_universe():
    """
    If none of the candidates are in the choice universe (or mapped), belief is undefined.
    """
    choice_map = make_choice_map_diapers()

    pivot_top = [
        {"token": ",", "logprob": 2.0},
        {"token": "Answer:", "logprob": 1.0},
    ]
    logprobs_content = make_logprobs_content_with_pivot(pivot_top, pivot_index=1)

    with pytest.raises(BeliefParseError, match="No valid logits within choice universe"):
        compute_belief(logprobs_content=logprobs_content, pivot_index=1, choice_map=choice_map)


def test_compute_belief_raises_on_out_of_range_pivot_index():
    choice_map = make_choice_map_diapers()
    logprobs_content = make_logprobs_content_with_pivot(
        [{"token": " P", "logprob": 0.0}],
        length=3,
        pivot_index=1,
    )

    with pytest.raises(BeliefInputError):
        compute_belief(logprobs_content=logprobs_content, pivot_index=10, choice_map=choice_map)


def test_compute_belief_raises_on_missing_logit_for_in_universe_token():
    """
    If a token is in-universe and mapped, but has missing logit, we fail fast.
    """
    choice_map = make_choice_map_diapers()

    pivot_top = [
        {"token": " P", "logprob": None},  # in-universe & mapped but missing
        {"token": " H", "logprob": 0.0},
    ]
    logprobs_content = make_logprobs_content_with_pivot(pivot_top, pivot_index=0)

    with pytest.raises(BeliefParseError, match=r"Invalid logit value.*None"):
        compute_belief(logprobs_content=logprobs_content, pivot_index=0, choice_map=choice_map)