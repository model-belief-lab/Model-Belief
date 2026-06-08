from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Set


# -----------------------------
# Choice-map config (diaper.yaml)
# -----------------------------

@dataclass(frozen=True)
class AlternativeConfig:
    id: str
    label: str
    target_tokens: List[str]


@dataclass(frozen=True)
class TokenPolicyConfig:
    case_sensitive: bool = True
    allow_leading_space_variant: bool = True


@dataclass(frozen=True)
class FirstTokenAvailableConfig:
    anchors: List[str] = field(default_factory=list)
    anchor_policy: Literal["first", "last"] = "last"


@dataclass(frozen=True)
class LlmJudgeConfig:
    # In your current diaper.yaml you still keep these here. We support it,
    # but you can also omit them and rely on models.yaml pivot_judge.
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_completion_tokens: Optional[int] = None
    return_format: Literal["json"] = "json"


@dataclass(frozen=True)
class CustomPivotConfig:
    function: str  # dotted path "pkg.module:callable"


@dataclass(frozen=True)
class PivotConfig:
    active: Literal["first_token_available", "llm_judge", "custom"]

    first_token_available: Optional[FirstTokenAvailableConfig] = None
    llm_judge: Optional[LlmJudgeConfig] = None
    custom: Optional[CustomPivotConfig] = None


@dataclass(frozen=True)
class ChoiceSetConfig:
    id: str
    alternatives: List[AlternativeConfig]


@dataclass(frozen=True)
class ChoiceMapConfig:
    version: int
    name: str
    description: str
    choice_set: ChoiceSetConfig
    token_policy: TokenPolicyConfig
    pivot: PivotConfig


# -----------------------------
# Models config (models.yaml)
# -----------------------------

@dataclass(frozen=True)
class OpenAIResponseConfig:
    model: str
    temperature: float
    max_completion_tokens: int
    logprobs: bool
    top_logprobs: int
    passthrough: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OpenAIPivotJudgeConfig:
    model: str
    temperature: float
    max_completion_tokens: int


@dataclass(frozen=True)
class OpenAIConfig:
    api_key_env: str = "OPENAI_API_KEY"
    response: OpenAIResponseConfig = None  # type: ignore[assignment]
    pivot_judge: OpenAIPivotJudgeConfig = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ModelsConfig:
    provider: Literal["openai"]
    openai: OpenAIConfig


# -----------------------------
# Derived structures
# -----------------------------

@dataclass(frozen=True)
class DerivedTokenMaps:
    """
    Derived token maps used by pivot + belief:

    - alt_to_tokens: alt_id -> set(tokens)
    - token_to_alt: token -> alt_id (must be 1-1)
    - token_universe: union of all alt tokens
    - normalized: whether tokens were normalized (e.g., lowercased)
    """
    alt_to_tokens: Mapping[str, Set[str]]
    token_to_alt: Mapping[str, str]
    token_universe: Set[str]
    normalized: bool


@dataclass(frozen=True)
class DerivedChoiceMap:
    """
    Wraps the raw ChoiceMapConfig plus derived token maps.
    """
    config: ChoiceMapConfig
    tokens: DerivedTokenMaps