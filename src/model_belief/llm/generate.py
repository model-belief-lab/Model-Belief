from __future__ import annotations

from typing import Any, Dict, Optional

from model_belief.config.loader import load_models
from .errors import LlmConfigError
from .types import LlmGenerationResult
from .openai_response import openai_generate_with_logprobs


def generate_with_logprobs(
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
    Provider-agnostic façade.

    Supports flexible input formats: either input/instructions or system/user prompts.
    Reads models.yaml or uses provided config to determine provider, then dispatches accordingly.
    """
    if models_config is not None:
        cfg = models_config
    elif models_yaml_path is not None:
        cfg = load_models(models_yaml_path)
    else:
        raise LlmConfigError("Provide either 'models_yaml_path' or 'models_config'.")

    provider = getattr(cfg, "provider", None)
    if not provider:
        raise LlmConfigError("models.yaml missing top-level 'provider'.")

    if provider == "openai":
        return openai_generate_with_logprobs(
            input=input,
            instructions=instructions,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            models_yaml_path=models_yaml_path,
            models_config=cfg,
            passthrough=passthrough,
            **overrides,
        )

    raise LlmConfigError(f"Unsupported provider: {provider!r}")