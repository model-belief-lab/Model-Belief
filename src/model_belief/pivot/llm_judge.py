from __future__ import annotations

import json
import os

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

from model_belief.config.models import DerivedChoiceMap

from .base import (
    PivotContext,
    PivotFinder,
    PivotInputError,
    PivotResult,
    token_from_logprob_item,
    validate_pivot_index,
)


API_KEY_FALLBACK = ""  # Optional: paste your OpenAI API key here. Prefer env var configured in models.yaml.

JudgeFn = Callable[..., int]
"""
A judge function is provided by the runtime script (not by pivot module).
It should return an integer pivot index into logprobs_content.

Recommended signature:
  judge_fn(
    text: str,
    tokens: Sequence[dict],     # [{index, token}] or richer
    token_universe: Sequence[str],
    choice_set_id: str,
    alternatives: Sequence[dict],  # [{id, label, target_tokens}] if useful
    metadata: dict | None,
  ) -> int
"""


# --- Helper functions for loading models config and OpenAI judge ---

def _get_models_config(ctx: PivotContext):
    from model_belief.config.loader import load_models

    if isinstance(ctx.metadata, dict) and ctx.metadata.get("models_config") is not None:
        return ctx.metadata["models_config"]

    if isinstance(ctx.metadata, dict) and ctx.metadata.get("models_yaml_path"):
        return load_models(ctx.metadata["models_yaml_path"])

    raise PivotInputError(
        "llm_judge: ctx.metadata must include either "
        "'models_config' (ModelsConfig) or "
        "'models_yaml_path' (path to models.yaml)."
    )


def _openai_judge_pivot_index(
    *,
    text: str,
    token_view: Sequence[dict],
    token_universe: Sequence[str],
    choice_set_id: str,
    alternatives: Sequence[dict],
    ctx: PivotContext,
) -> int:
    models_config = _get_models_config(ctx)

    api_key_env = models_config.openai.api_key_env
    api_key = os.getenv(api_key_env) or API_KEY_FALLBACK
    if not api_key:
        raise PivotInputError(
            f"llm_judge: OpenAI API key not found. Set env var '{api_key_env}' or set API_KEY_FALLBACK in pivot/llm_judge.py."
        )

    try:
        from openai import OpenAI
    except Exception as e:
        raise PivotInputError(
            "llm_judge: OpenAI SDK not available. Install it with 'pip install openai'."
        ) from e

    client = OpenAI(api_key=api_key)

    judge_cfg = models_config.openai.pivot_judge

    system_prompt = (
        "You are a strict evaluator. Your job is to identify the pivot token index in a tokenized model output. "
        "The pivot token is the FIRST token whose realization uniquely determines the model's choice among the provided alternatives. "
        "You MUST base your decision ONLY on the provided 'tokens' sequence and 'token_universe'. "
        "Return STRICT JSON only, exactly in the form: {\"pivot_index\": <int> }."
    )

    user_prompt = json.dumps(
        {
            "choice_set_id": choice_set_id,
            "alternatives": alternatives,
            "token_universe": list(token_universe),
            "tokens": list(token_view),
        },
        ensure_ascii=False,
    )

    if isinstance(ctx.metadata, dict):
        ctx.metadata["_openai_judge_trace"] = {
            "judge_model": judge_cfg.model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "included_output_text": False,
        }

    resp = client.responses.create(
        model=judge_cfg.model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_output_tokens=judge_cfg.max_completion_tokens,
        reasoning={"effort": "minimal"},
    )

    # Extract text output robustly
    out_text: Optional[str] = None

    # 1) Preferred: SDK convenience
    try:
        out_text = getattr(resp, "output_text", None)
        if isinstance(out_text, str):
            out_text = out_text.strip() or None
    except Exception:
        out_text = None

    # 2) Fallback: inspect output items (works for dict or object forms)
    if not out_text:
        try:
            raw = None
            if hasattr(resp, "model_dump"):
                raw = resp.model_dump()
            elif hasattr(resp, "dict"):
                raw = resp.dict()
            else:
                raw = resp  # last resort

            # Save raw response for debugging (best-effort)
            if isinstance(ctx.metadata, dict):
                trace = ctx.metadata.get("_openai_judge_trace")
                if isinstance(trace, dict):
                    trace["raw_response_json"] = raw if isinstance(raw, (dict, list)) else str(raw)

            # Normalize output list
            output_items = None
            if isinstance(raw, dict):
                output_items = raw.get("output")
            else:
                output_items = getattr(resp, "output", None)

            parts: list[str] = []
            if output_items:
                for item in output_items:
                    # item may be dict or object
                    content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
                    if not content:
                        continue
                    for c in content:
                        c_type = c.get("type") if isinstance(c, dict) else getattr(c, "type", None)

                        # Standard text
                        if c_type in ("output_text", "text"):
                            t = c.get("text") if isinstance(c, dict) else getattr(c, "text", "")
                            if t:
                                parts.append(t)

                        # Sometimes JSON is returned as structured content
                        elif c_type in ("output_json", "json"):
                            j = c.get("json") if isinstance(c, dict) else getattr(c, "json", None)
                            if j is not None:
                                parts.append(json.dumps(j, ensure_ascii=False))

            out_text = "".join(parts).strip() or None
        except Exception:
            out_text = None

    if not out_text:
        raise PivotInputError("llm_judge: judge model returned no text output.")

    if isinstance(ctx.metadata, dict):
        trace = ctx.metadata.get("_openai_judge_trace")
        if isinstance(trace, dict):
            trace["raw_output_text"] = out_text

    try:
        data = json.loads(out_text)
    except Exception as e:
        raise PivotInputError(
            f"llm_judge: failed to parse judge output as JSON. Output was: {out_text!r}"
        ) from e

    if not isinstance(data, dict) or "pivot_index" not in data:
        raise PivotInputError(
            f"llm_judge: judge output JSON must contain 'pivot_index'. Output was: {out_text!r}"
        )

    idx = data["pivot_index"]
    if not isinstance(idx, int):
        raise PivotInputError(
            f"llm_judge: 'pivot_index' must be an int. Got: {idx!r}"
        )

    return idx


@dataclass(frozen=True)
class LlmJudgeFinder:
    """
    Strategy: llm_judge

    By default, this module uses an external judge function (ctx.judge_fn) if supplied.
    If not supplied, it will run an internal OpenAI-based judge using parameters from models.yaml.
    The judge must return a pivot index. We then run strict validation:
      - in range
      - token at index ∈ token_universe
      - token maps to exactly one alternative
    """
    # Optional: provide a compact token list to the judge (limits prompt size)
    max_tokens_to_show: int = 256

    def __call__(
        self,
        logprobs_content: Sequence[Any],
        choice_map: DerivedChoiceMap,
        ctx: PivotContext,
    ) -> PivotResult:
        method = "llm_judge"
        if not ctx.text:
            raise PivotInputError(f"{method}: ctx.text is required.")
        # Prepare a compact token view for the judge (index + token string).
        # V1: we provide the first N tokens; you can change this in runtime if needed.
        token_view = [{"index": i, "token": token_from_logprob_item(it)} for i, it in enumerate(logprobs_content[: self.max_tokens_to_show])]

        alternatives_view = [
            {"id": alt.id, "label": alt.label, "target_tokens": alt.target_tokens}
            for alt in choice_map.config.choice_set.alternatives
        ]

        used_openai_judge = False
        judge_model = None
        if ctx.judge_fn:
            idx = ctx.judge_fn(
                text=ctx.text,
                tokens=token_view,
                token_universe=sorted(choice_map.tokens.token_universe),
                choice_set_id=choice_map.config.choice_set.id,
                alternatives=alternatives_view,
                metadata=ctx.metadata,
            )
        else:
            idx = _openai_judge_pivot_index(
                text=ctx.text,
                token_view=token_view,
                token_universe=sorted(choice_map.tokens.token_universe),
                choice_set_id=choice_map.config.choice_set.id,
                alternatives=alternatives_view,
                ctx=ctx,
            )
            used_openai_judge = True
            models_config = _get_models_config(ctx)
            judge_model = getattr(models_config.openai.pivot_judge, "model", None)

        res = validate_pivot_index(idx, logprobs_content, choice_map, method=method)
        debug_fields = {**res.debug, "judge_max_tokens_to_show": self.max_tokens_to_show, "used_openai_judge": used_openai_judge}
        if used_openai_judge:
            debug_fields["judge_model"] = judge_model
        return PivotResult(
            index=res.index,
            token=res.token,
            choice_id=res.choice_id,
            method=res.method,
            debug=debug_fields,
        )