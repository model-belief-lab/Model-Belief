from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from model_belief.config.models import DerivedChoiceMap
from model_belief.llm.types import token_from_logprob_item


class PivotError(ValueError):
    pass


class PivotValidationError(PivotError):
    pass


class PivotInputError(PivotError):
    pass


def _is_mapping(x: Any) -> bool:
    return isinstance(x, dict)


def normalize_token(token: str, *, case_sensitive: bool) -> str:
    return token if case_sensitive else token.lower()

def find_anchor_span(text: str, anchors: Sequence[str], anchor_policy: str) -> Optional[Tuple[int, int]]:
    """
    Return the (start, end) span of the chosen anchor occurrence in text.
    anchor_policy: "first" or "last"
    """
    if not anchors:
        return None
    positions: List[Tuple[int, int]] = []
    for a in anchors:
        if not a:
            continue
        start = 0
        while True:
            idx = text.find(a, start)
            if idx == -1:
                break
            positions.append((idx, idx + len(a)))
            start = idx + len(a)

    if not positions:
        return None

    if anchor_policy == "first":
        return positions[0]
    if anchor_policy == "last":
        return positions[-1]
    raise PivotInputError(f"Invalid anchor_policy: {anchor_policy}. Expected 'first' or 'last'.")


@dataclass(frozen=True)
class PivotResult:
    index: int
    token: str
    choice_id: str
    method: str
    debug: Dict[str, Any]


@dataclass(frozen=True)
class PivotContext:
    """
    Extra runtime context optionally used by pivot strategies.
    - text: the model output text (required if using anchor-based strategy)
    - judge_fn: callable used by llm_judge strategy
    """
    text: Optional[str] = None
    judge_fn: Optional[Callable[..., int]] = None
    # Optional extra info the user may want to provide to judge/custom strategies
    metadata: Optional[Dict[str, Any]] = None


class PivotFinder(Protocol):
    def __call__(
        self,
        logprobs_content: Sequence[Any],
        choice_map: DerivedChoiceMap,
        ctx: PivotContext,
    ) -> PivotResult:
        ...


def validate_pivot_index(
    index: int,
    logprobs_content: Sequence[Any],
    choice_map: DerivedChoiceMap,
    *,
    method: str,
) -> PivotResult:
    """
    Shared post-validation for all strategies:
      - index in bounds
      - token belongs to token_universe
      - token maps to exactly one alternative (token_to_alt is 1-1 by construction)
    """
    if not isinstance(index, int):
        raise PivotValidationError(f"{method}: pivot index must be int, got {type(index)}")

    if index < 0 or index >= len(logprobs_content):
        raise PivotValidationError(f"{method}: pivot index {index} out of range [0, {len(logprobs_content)-1}]")

    raw_token = token_from_logprob_item(logprobs_content[index])

    # Normalize token exactly the same way as config derivation did.
    case_sensitive = choice_map.config.token_policy.case_sensitive
    tok = normalize_token(raw_token, case_sensitive=case_sensitive)

    if tok not in choice_map.tokens.token_universe:
        raise PivotValidationError(
            f"{method}: token at pivot index not in token_universe. "
            f"index={index}, token={raw_token!r}"
        )

    # token_to_alt already guaranteed 1-1 by derive.py; still guard.
    if tok not in choice_map.tokens.token_to_alt:
        raise PivotValidationError(
            f"{method}: token is in universe but missing token_to_alt mapping. token={raw_token!r}"
        )

    choice_id = choice_map.tokens.token_to_alt[tok]
    return PivotResult(
        index=index,
        token=raw_token,
        choice_id=choice_id,
        method=method,
        debug={"normalized_token": tok},
    )


def resolve_pivot(
    finder: PivotFinder,
    logprobs_content: Sequence[Any],
    choice_map: DerivedChoiceMap,
    ctx: Optional[PivotContext] = None,
) -> PivotResult:
    """
    Primary entry point:
      - runs the strategy-specific finder
      - returns a validated PivotResult
    """
    ctx = ctx or PivotContext()
    return finder(logprobs_content, choice_map, ctx)