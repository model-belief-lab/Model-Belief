from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .errors import LlmConfigError, LlmProviderError
from .types import LlmGenerationResult, coerce_passthrough
from .openai_client import get_openai_client


def _best_effort_dump(resp: Any) -> Any:
    """Best-effort JSON-ish dump of SDK response for debugging."""
    try:
        if hasattr(resp, "model_dump"):
            return resp.model_dump()
    except Exception:
        pass
    try:
        if hasattr(resp, "dict"):
            return resp.dict()
    except Exception:
        pass
    return str(resp)


def _extract_text(resp: Any) -> str:
    """
    Best-effort extraction of output text from Responses API response.
    """
    # SDK convenience
    try:
        t = getattr(resp, "output_text", None)
        if isinstance(t, str) and t.strip():
            return t.strip()
    except Exception:
        pass

    # Fallback: scan output blocks
    try:
        parts: List[str] = []
        for item in getattr(resp, "output", []) or []:
            for block in getattr(item, "content", []) or []:
                if getattr(block, "type", None) in ("output_text", "text"):
                    txt = getattr(block, "text", "")
                    if txt:
                        parts.append(txt)
        return "".join(parts).strip()
    except Exception:
        return ""


def _extract_logprobs_content(resp: Any) -> Sequence[Any]:
    """
    Extract token-level logprobs list used by pivot + belief.

    Contract:
      - Returned sequence is aligned to the generated output_text token stream.
      - Each element is typically a dict/object containing:
          token: str
          logprob: float (score)
          top_logprobs: optional list[ {token, logprob, ...} ]
    """
    # Preferred: scan assistant message output_text blocks
    try:
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", None) != "message":
                continue
            for block in getattr(item, "content", []) or []:
                if getattr(block, "type", None) != "output_text":
                    continue
                lp = getattr(block, "logprobs", None)
                if isinstance(lp, (list, tuple)) and len(lp) > 0:
                    return lp
    except Exception:
        pass

    # Fallback: some SDK shapes might attach logprobs at top-level (rare)
    try:
        lp = getattr(resp, "logprobs", None)
        if isinstance(lp, (list, tuple)) and len(lp) > 0:
            return lp
    except Exception:
        pass

    return []


def _build_input_and_instructions(
    *,
    input: Any | None,
    instructions: str | None,
    system_prompt: str | None,
    user_prompt: str | None,
) -> Tuple[Any, Optional[str]]:
    """Build Responses API `input` and `instructions`.

    Precedence:
      1) If `input` is provided, use it as-is. If `instructions` is provided, pass it through.
      2) Else, require `user_prompt`.
         - If `system_prompt` is provided and `instructions` is not, map it to `instructions`.
         - Use `user_prompt` as the `input` string.
    """
    if input is not None:
        return input, instructions
    if not user_prompt:
        raise LlmConfigError("OpenAI response input missing: provide either `input` or `user_prompt`.")
    final_instructions = instructions if instructions is not None else system_prompt
    return user_prompt, final_instructions


def openai_generate_with_logprobs(
    *,
    input: Any | None = None,
    instructions: str | None = None,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    models_yaml_path: str | None = None,
    models_config: Any | None = None,
    passthrough: Optional[Dict[str, Any]] = None,
    **overrides: Any,
) -> LlmGenerationResult:
    """
    Main generation call (Responses API) used to produce output + logprobs.

    Supports flexible input options:

      - Provide `input` directly as the Responses API input payload.
      - Optionally provide `instructions` alongside `input`.
      - Alternatively, provide `user_prompt` string.
        If `system_prompt` is provided and `instructions` is not, `system_prompt` will be used as `instructions`.
      - `input` and `instructions` take precedence over `system_prompt` and `user_prompt`.

    Defaults come from models.yaml:
      openai.response.{model, temperature?, max_completion_tokens, logprobs, top_logprobs, passthrough}

    - 'passthrough' lets user forward arbitrary Responses API parameters.
    - **overrides take highest priority**.
    """
    client, cfg = get_openai_client(models_yaml_path=models_yaml_path, models_config=models_config)

    openai_cfg = getattr(cfg, "openai", None)
    if openai_cfg is None:
        raise LlmConfigError("models_config.openai missing.")

    response_cfg = getattr(openai_cfg, "response", None)
    if response_cfg is None:
        raise LlmConfigError("models_config.openai.response missing (expected 'response:' in models.yaml).")

    model = getattr(response_cfg, "model", None)
    if not model:
        raise LlmConfigError("openai.response.model missing.")

    max_completion_tokens = getattr(response_cfg, "max_completion_tokens", None)
    want_logprobs = getattr(response_cfg, "logprobs", None)
    top_logprobs = getattr(response_cfg, "top_logprobs", None)

    if want_logprobs is not True:
        raise LlmConfigError("openai.response.logprobs must be true for model-belief extraction.")
    if not isinstance(top_logprobs, int) or top_logprobs <= 0:
        raise LlmConfigError("openai.response.top_logprobs must be a positive integer.")

    base_passthrough = coerce_passthrough(getattr(response_cfg, "passthrough", None))
    runtime_passthrough = coerce_passthrough(passthrough)
    merged_passthrough = {**base_passthrough, **runtime_passthrough}

    input_payload, final_instructions = _build_input_and_instructions(
        input=input,
        instructions=instructions,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    # Build request (ONLY include params that matter + safe defaults).
    # Logprobs are requested via 'include' parameter, not a boolean 'logprobs' flag.
    req: Dict[str, Any] = {
        "model": model,
        "input": input_payload,
        "include": ["message.output_text.logprobs"],
        "top_logprobs": top_logprobs,
    }
    if final_instructions is not None and final_instructions != "":
        req["instructions"] = final_instructions

    if max_completion_tokens is not None:
        req["max_output_tokens"] = int(max_completion_tokens)

    # Temperature is OPTIONAL. Some models may reject it.
    # Include only if present in config AND not overridden/passthrough.
    temperature = getattr(response_cfg, "temperature", None)
    if temperature is not None and "temperature" not in merged_passthrough and "temperature" not in overrides:
        req["temperature"] = float(temperature)

    # Apply passthrough then overrides (overrides win).
    req.update(merged_passthrough)
    req.update(overrides)

    try:
        resp = client.responses.create(**req)
    except Exception as e:
        raise LlmProviderError(f"OpenAI responses.create failed: {e}") from e

    text = _extract_text(resp)
    logprobs_content = _extract_logprobs_content(resp)

    has_top = False
    try:
        if logprobs_content and isinstance(logprobs_content, (list, tuple)):
            first = logprobs_content[0]
            if isinstance(first, dict):
                has_top = isinstance(first.get("top_logprobs"), list)
            else:
                has_top = hasattr(first, "top_logprobs")
    except Exception:
        has_top = False

    debug = {
        "provider": "openai",
        "model": model,
        "request_parameters": req,
        "has_text": bool(text),
        "logprobs_len": len(logprobs_content) if hasattr(logprobs_content, "__len__") else None,
        "logprobs_has_top_logprobs": has_top,
        "logprobs_item_example": logprobs_content[0] if logprobs_content else None,
        "raw_response_dump": _best_effort_dump(resp),
    }

    return LlmGenerationResult(text=text, logprobs_content=logprobs_content, raw=resp, debug=debug)