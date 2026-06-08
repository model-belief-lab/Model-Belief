from __future__ import annotations

import os
from typing import Any, Tuple

from model_belief.config.loader import load_models
from .errors import LlmConfigError, LlmProviderError


def get_openai_client(
    *,
    models_yaml_path: str | None = None,
    models_config: Any | None = None,
) -> Tuple[Any, Any]:
    """
    Return (OpenAI_client, models_config).

    One of models_yaml_path or models_config must be provided.
    Reads OpenAI API key from env specified by models_config.openai.api_key_env.
    """
    if models_config is None:
        if not models_yaml_path:
            raise LlmConfigError("OpenAI client: provide models_yaml_path or models_config.")
        models_config = load_models(models_yaml_path)

    openai_cfg = getattr(models_config, "openai", None)
    if openai_cfg is None:
        raise LlmConfigError("OpenAI client: models_config.openai is missing.")

    api_key_env = getattr(openai_cfg, "api_key_env", None)
    if not api_key_env:
        raise LlmConfigError("OpenAI client: openai.api_key_env is missing in models config.")

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise LlmConfigError(f"OpenAI client: API key not found. Set env var '{api_key_env}'.")

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise LlmProviderError("OpenAI SDK not available. Install: pip install openai") from e

    return OpenAI(api_key=api_key), models_config