from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class LlmGenerationResult:
    """
    Standard result returned by llm generation functions.

    - text: model output text (best-effort)
    - logprobs_content: token-level objects as returned by provider (or normalized dicts)
    - raw: raw provider response (for debugging)
    """
    text: str
    logprobs_content: Sequence[Any]
    raw: Any
    debug: Dict[str, Any]


def coerce_passthrough(passthrough: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a shallow copy of passthrough dict, or empty dict."""
    if not passthrough:
        return {}
    if not isinstance(passthrough, dict):
        raise TypeError(f"passthrough must be a dict, got {type(passthrough)}")
    return dict(passthrough)

def token_from_logprob_item(item: Any) -> str:
    """Return realized token string from a token-level logprob item."""
    if isinstance(item, dict):
        t = item.get("token")
        return t if isinstance(t, str) else ""
    return getattr(item, "token", "") or ""


def top_logprobs_from_logprob_item(item: Any) -> List[Dict[str, Any]]:
    """Return top_logprobs list (candidate tokens) from a token-level item."""
    if isinstance(item, dict):
        tlp = item.get("top_logprobs")
        return tlp if isinstance(tlp, list) else []
    tlp = getattr(item, "top_logprobs", None)
    return tlp if isinstance(tlp, list) else []


def realized_tokens(logprobs_content: Sequence[Any]) -> List[str]:
    """Token list aligned to the generated output text."""
    return [token_from_logprob_item(it) for it in (logprobs_content or [])]