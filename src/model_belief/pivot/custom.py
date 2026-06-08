from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence, Tuple

from model_belief.config.models import DerivedChoiceMap

from .base import (
    PivotContext,
    PivotFinder,
    PivotInputError,
    PivotResult,
    validate_pivot_index,
)


def load_dotted_callable(path: str) -> Callable[..., Any]:
    """
    Load a callable from 'pkg.module:func' dotted path.
    """
    if ":" not in path:
        raise PivotInputError(f"Custom pivot function must be in 'module:callable' format, got: {path}")
    mod_name, fn_name = path.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name, None)
    if fn is None or not callable(fn):
        raise PivotInputError(f"Cannot load callable '{fn_name}' from module '{mod_name}'")
    return fn


@dataclass(frozen=True)
class CustomFinder:
    function_path: str

    def __call__(
        self,
        logprobs_content: Sequence[Any],
        choice_map: DerivedChoiceMap,
        ctx: PivotContext,
    ) -> PivotResult:
        method = "custom"
        fn = load_dotted_callable(self.function_path)

        # Provide a stable context dictionary for custom logic.
        context = {
            "token_universe": choice_map.tokens.token_universe,
            "token_to_alt": choice_map.tokens.token_to_alt,
            "alt_to_tokens": choice_map.tokens.alt_to_tokens,
            "choice_set_id": choice_map.config.choice_set.id,
            "alternatives": [
                {"id": alt.id, "label": alt.label, "target_tokens": alt.target_tokens}
                for alt in choice_map.config.choice_set.alternatives
            ],
            "token_policy": {
                "case_sensitive": choice_map.config.token_policy.case_sensitive,
                "allow_leading_space_variant": choice_map.config.token_policy.allow_leading_space_variant,
            },
            "metadata": ctx.metadata or {},
        }

        idx = fn(text=ctx.text, logprobs_content=logprobs_content, context=context)
        res = validate_pivot_index(idx, logprobs_content, choice_map, method=method)
        return PivotResult(
            index=res.index,
            token=res.token,
            choice_id=res.choice_id,
            method=res.method,
            debug={**res.debug, "custom_function": self.function_path},
        )