from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from model_belief.config.models import DerivedChoiceMap

from .base import (
    PivotContext,
    PivotFinder,
    PivotInputError,
    PivotResult,
    find_anchor_span,
    normalize_token,
    token_from_logprob_item,
    validate_pivot_index,
)


@dataclass(frozen=True)
class FirstTokenAvailableFinder:
    """
    Strategy: first_token_available

    If anchors are provided, requires ctx.text, and selects the first token in the output
    after the chosen anchor occurrence whose token is in token_universe.

    Note: We do not try to align character offsets to token offsets. In V1 we simply
    restrict search by *presence* of anchor and then take the first universe token in
    the entire generation. The anchor primarily prevents early mention of choice tokens
    when prompts enforce an "Answer:" section.

    If you need precise alignment later, we can add token offset tracking in the llm wrapper.
    """
    anchors: Sequence[str]
    anchor_policy: str  # "first" | "last"

    def __call__(
        self,
        logprobs_content: Sequence[Any],
        choice_map: DerivedChoiceMap,
        ctx: PivotContext,
    ) -> PivotResult:
        method = "first_token_available"

        # If anchors configured, require text and at least one anchor occurrence.
        if self.anchors:
            if not ctx.text:
                raise PivotInputError(f"{method}: anchors configured but ctx.text is missing.")
            span = find_anchor_span(ctx.text, self.anchors, self.anchor_policy)
            if span is None:
                raise PivotInputError(f"{method}: none of the anchors found in ctx.text.")
            # V1 behavior: anchors gate correctness conceptually; we cannot map span->token index
            # without offsets. We record it for debugging.
            debug_anchor = {"anchor_span": span, "anchors": list(self.anchors), "anchor_policy": self.anchor_policy}
        else:
            debug_anchor = {}

        case_sensitive = choice_map.config.token_policy.case_sensitive

        # Find first token that belongs to token_universe
        for i, item in enumerate(logprobs_content):
            raw_tok = token_from_logprob_item(item)
            tok = normalize_token(raw_tok, case_sensitive=case_sensitive)
            if tok in choice_map.tokens.token_universe:
                res = validate_pivot_index(i, logprobs_content, choice_map, method=method)
                # Attach anchor debug info
                return PivotResult(
                    index=res.index,
                    token=res.token,
                    choice_id=res.choice_id,
                    method=res.method,
                    debug={**res.debug, **debug_anchor},
                )

        raise PivotInputError(f"{method}: no token from token_universe found in the generation.")