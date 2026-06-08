from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

from model_belief.config.models import DerivedChoiceMap

from .errors import BeliefInputError, BeliefParseError
from .types import Belief, BeliefItem


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read dict key or attribute."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _logprob_to_prob(logit: Any) -> float:
    if logit is None:
        raise BeliefParseError("Missing logit value.")
    try:
        lp = float(logit)
    except Exception as e:
        raise BeliefParseError(f"Invalid logit value: {logit!r}") from e
    return float(math.exp(lp))


def compute_belief(
    *,
    logprobs_content: Sequence[Any],
    pivot_index: int,
    choice_map: DerivedChoiceMap,
) -> Belief:
    """
    Compute belief (sub-distribution) over alternatives at a pivot position.

    Inputs:
      - logprobs_content: token-level logprob sequence (provider-native objects or dicts).
      - pivot_index: index into logprobs_content for the pivot token.
      - choice_map: DerivedChoiceMap containing token_universe and token_to_alt mapping.

    Assumptions:
      - At logprobs_content[pivot_index], we can read a list of candidate tokens under:
          item.top_logprobs  (preferred)
        Each candidate contains:
          cand.token (str), cand.logprob (float)
      - We compute a numerically stable softmax over logits of candidate tokens that map into alternatives.

    Notes:
      - tokens not in token_universe (or unmapped) are ignored and logged in debug.
    """
    if pivot_index < 0 or pivot_index >= len(logprobs_content):
        raise BeliefInputError(f"pivot_index out of range: {pivot_index} (len={len(logprobs_content)})")

    item = logprobs_content[pivot_index]

    top = _get(item, "top_logprobs", None)
    if top is None:
        # Some providers might put candidates under another field; keep minimal fallback
        top = _get(item, "logprobs", None)

    if not isinstance(top, (list, tuple)) or len(top) == 0:
        raise BeliefParseError(
            "No top_logprobs found at pivot_index. "
            "Ensure OpenAI Responses request includes include=['message.output_text.logprobs'] and top_logprobs>0."
        )

    token_universe = choice_map.tokens.token_universe
    token_to_alt = choice_map.tokens.token_to_alt

    used: List[Tuple[str, float, str]] = []
    ignored: List[Tuple[str, float, str]] = []
    logits: List[Tuple[str, float]] = []

    num_candidates_outside_universe = 0

    for cand in top:
        tok = _get(cand, "token", None)
        score = _get(cand, "logprob", None)
        if not isinstance(tok, str) or tok == "":
            continue

        if tok not in token_universe:
            ignored.append((tok, score if score is not None else float("nan"), "not_in_token_universe"))
            num_candidates_outside_universe += 1
            continue

        alt_id = token_to_alt.get(tok)
        if alt_id is None:
            ignored.append((tok, score if score is not None else float("nan"), "no_alt_mapping"))
            num_candidates_outside_universe += 1
            continue

        try:
            score_float = float(score)
        except Exception as e:
            raise BeliefParseError(f"Invalid logit value for token {tok!r}: {score!r}") from e

        used.append((tok, score_float, alt_id))
        logits.append((alt_id, score_float))

    if len(logits) == 0:
        raise BeliefParseError("No valid logits within choice universe at pivot_index.")

    # Compute numerically stable softmax over logits
    max_logit = max(score for _, score in logits)
    exp_weights = [(alt_id, math.exp(score - max_logit)) for alt_id, score in logits]
    Z = sum(w for _, w in exp_weights)

    prob_by_alt: Dict[str, float] = {}
    for alt_id, w in exp_weights:
        prob_by_alt[alt_id] = prob_by_alt.get(alt_id, 0.0) + w / Z

    items = [BeliefItem(alt_id=alt_id, prob=prob) for alt_id, prob in prob_by_alt.items()]
    items.sort(key=lambda it: it.prob, reverse=True)

    debug: Dict[str, object] = {
        "pivot_index": pivot_index,
        "top_k": len(top),
        "num_candidates_in_universe": len(logits),
        "num_candidates_outside_universe": num_candidates_outside_universe,
        "used_tokens": used,
        "ignored_tokens": ignored,
        "normalization": "softmax_over_choice_universe",
    }

    return Belief(
        items=items,
        total_mass=1.0,
        missing_mass=0.0,
        pivot_index=pivot_index,
        debug=debug,
    )