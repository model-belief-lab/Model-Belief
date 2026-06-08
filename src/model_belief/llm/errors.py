from __future__ import annotations


class LlmError(RuntimeError):
    """Base error for LLM calls."""


class LlmConfigError(LlmError):
    """Raised when models config is missing or invalid."""


class LlmProviderError(LlmError):
    """Raised when provider SDK is missing or provider call fails."""