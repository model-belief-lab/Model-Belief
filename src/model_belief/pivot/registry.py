from __future__ import annotations

from typing import Optional

from model_belief.config.models import DerivedChoiceMap

from .base import PivotFinder
from .custom import CustomFinder
from .first_available import FirstTokenAvailableFinder
from .llm_judge import LlmJudgeFinder


def get_pivot_finder(choice_map: DerivedChoiceMap) -> PivotFinder:
    """
    Instantiate the active pivot strategy based on choice_map.config.pivot.active.

    Note: llm_judge requires ctx.judge_fn at runtime.
    """
    active = choice_map.config.pivot.active

    if active == "first_token_available":
        cfg = choice_map.config.pivot.first_token_available
        # loader enforces this block exists for active strategy
        return FirstTokenAvailableFinder(
            anchors=cfg.anchors,
            anchor_policy=cfg.anchor_policy,
        )

    if active == "llm_judge":
        # You can tune max_tokens_to_show later; keep V1 conservative.
        return LlmJudgeFinder(max_tokens_to_show=256)

    if active == "custom":
        cfg = choice_map.config.pivot.custom
        return CustomFinder(function_path=cfg.function)

    raise ValueError(f"Unknown pivot strategy: {active}")