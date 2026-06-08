

"""
LLM provider interfaces and generation utilities.

This subpackage provides:
- client abstractions (e.g., OpenAI)
- response parsing helpers
- high-level generation entrypoints

The design keeps provider-specific logic isolated while exposing
stable, research-friendly APIs at the package level.
"""

from .errors import (
    LlmConfigError,
    LlmProviderError,
    LlmError,
)
from .types import (
    LlmGenerationResult,
)

from .openai_client import get_openai_client
from .openai_response import openai_generate_with_logprobs
from .generate import generate_with_logprobs
from .types import realized_tokens, token_from_logprob_item, top_logprobs_from_logprob_item

__all__ = [
    "LlmConfigError",
    "LlmProviderError",
    "LlmError",
    "LlmGenerationResult",
    "get_openai_client",
    "openai_generate_with_logprobs",
    "generate_with_logprobs",
    "realized_tokens",
    "token_from_logprob_item",
    "top_logprobs_from_logprob_item",
]