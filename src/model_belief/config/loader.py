from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

from .derive import ConfigDerivationError, derive_token_maps
from .models import (
    AlternativeConfig,
    ChoiceMapConfig,
    ChoiceSetConfig,
    CustomPivotConfig,
    DerivedChoiceMap,
    FirstTokenAvailableConfig,
    LlmJudgeConfig,
    ModelsConfig,
    OpenAIConfig,
    OpenAIPivotJudgeConfig,
    OpenAIResponseConfig,
    PivotConfig,
    TokenPolicyConfig,
)


class ConfigError(ValueError):
    pass


def _read_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"YAML root must be a mapping/dict: {p}")
    return data


def _require(d: Mapping[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise ConfigError(f"Missing required key '{key}' in {ctx}")
    return d[key]


def _as_dict(x: Any, ctx: str) -> Dict[str, Any]:
    if not isinstance(x, dict):
        raise ConfigError(f"Expected a mapping/dict in {ctx}, got {type(x)}")
    return x


def _as_list(x: Any, ctx: str) -> List[Any]:
    if not isinstance(x, list):
        raise ConfigError(f"Expected a list in {ctx}, got {type(x)}")
    return x


def load_choice_map(path: str | Path) -> DerivedChoiceMap:
    """
    Load diaper-like choice-map YAML and derive token maps (universe, token_to_alt, etc.).
    """
    raw = _read_yaml(path)

    version = int(_require(raw, "version", "choice_map"))
    name = str(_require(raw, "name", "choice_map"))
    description = str(_require(raw, "description", "choice_map"))

    choice_set_raw = _as_dict(_require(raw, "choice_set", "choice_map"), "choice_map.choice_set")
    choice_set_id = str(_require(choice_set_raw, "id", "choice_map.choice_set"))

    alts_raw = _as_list(_require(choice_set_raw, "alternatives", "choice_map.choice_set"), "choice_map.choice_set.alternatives")
    alternatives: List[AlternativeConfig] = []
    seen_alt_ids = set()

    for i, a in enumerate(alts_raw):
        a_dict = _as_dict(a, f"choice_map.choice_set.alternatives[{i}]")
        alt_id = str(_require(a_dict, "id", f"alternative[{i}]"))
        if alt_id in seen_alt_ids:
            raise ConfigError(f"Duplicate alternative id '{alt_id}'")
        seen_alt_ids.add(alt_id)

        label = str(_require(a_dict, "label", f"alternative[{i}]"))
        toks = _as_list(_require(a_dict, "target_tokens", f"alternative[{i}]"), f"alternative[{i}].target_tokens")
        target_tokens = [str(t) for t in toks]

        alternatives.append(AlternativeConfig(id=alt_id, label=label, target_tokens=target_tokens))

    token_policy_raw = _as_dict(_require(raw, "token_policy", "choice_map"), "choice_map.token_policy")
    token_policy = TokenPolicyConfig(
        case_sensitive=bool(token_policy_raw.get("case_sensitive", True)),
        allow_leading_space_variant=bool(token_policy_raw.get("allow_leading_space_variant", True)),
    )

    pivot_raw = _as_dict(_require(raw, "pivot", "choice_map"), "choice_map.pivot")
    active = str(_require(pivot_raw, "active", "choice_map.pivot"))
    if active not in {"first_token_available", "llm_judge", "custom"}:
        raise ConfigError(f"pivot.active must be one of first_token_available|llm_judge|custom, got '{active}'")

    # first_token_available (optional unless active)
    fta_cfg: Optional[FirstTokenAvailableConfig] = None
    if "first_token_available" in pivot_raw:
        fta_raw = _as_dict(pivot_raw["first_token_available"], "choice_map.pivot.first_token_available")
        anchors = [str(x) for x in fta_raw.get("anchors", [])] if isinstance(fta_raw.get("anchors", []), list) else None
        if anchors is None:
            raise ConfigError("pivot.first_token_available.anchors must be a list of strings")
        anchor_policy = str(fta_raw.get("anchor_policy", "last"))
        if anchor_policy not in {"first", "last"}:
            raise ConfigError("pivot.first_token_available.anchor_policy must be 'first' or 'last'")
        fta_cfg = FirstTokenAvailableConfig(anchors=anchors, anchor_policy=anchor_policy)  # type: ignore[arg-type]

    # llm_judge (optional unless active)
    judge_cfg: Optional[LlmJudgeConfig] = None
    if "llm_judge" in pivot_raw:
        lj_raw = _as_dict(pivot_raw["llm_judge"], "choice_map.pivot.llm_judge")
        judge_cfg = LlmJudgeConfig(
            model=lj_raw.get("model"),
            temperature=lj_raw.get("temperature"),
            max_completion_tokens=lj_raw.get("max_completion_tokens"),
            return_format=str(lj_raw.get("return_format", "json")),
        )

    # custom (optional unless active)
    custom_cfg: Optional[CustomPivotConfig] = None
    if "custom" in pivot_raw:
        c_raw = _as_dict(pivot_raw["custom"], "choice_map.pivot.custom")
        fn = str(_require(c_raw, "function", "choice_map.pivot.custom"))
        custom_cfg = CustomPivotConfig(function=fn)

    # Validate active strategy block exists
    if active == "first_token_available" and fta_cfg is None:
        raise ConfigError("pivot.active=first_token_available requires pivot.first_token_available block")
    if active == "llm_judge" and judge_cfg is None:
        raise ConfigError("pivot.active=llm_judge requires pivot.llm_judge block")
    if active == "custom" and custom_cfg is None:
        raise ConfigError("pivot.active=custom requires pivot.custom block")

    pivot = PivotConfig(
        active=active,  # type: ignore[arg-type]
        first_token_available=fta_cfg,
        llm_judge=judge_cfg,
        custom=custom_cfg,
    )

    cfg = ChoiceMapConfig(
        version=version,
        name=name,
        description=description,
        choice_set=ChoiceSetConfig(id=choice_set_id, alternatives=alternatives),
        token_policy=token_policy,
        pivot=pivot,
    )

    # Derive token maps (universe, token_to_alt, etc.)
    try:
        derived = derive_token_maps(cfg.choice_set.alternatives, cfg.token_policy)
    except ConfigDerivationError as e:
        raise ConfigError(str(e)) from e

    return DerivedChoiceMap(config=cfg, tokens=derived)


def load_models(path: str | Path) -> ModelsConfig:
    """
    Load models.yaml, supporting:
      provider: openai
      openai.api_key_env
      openai.response.{model, temperature, max_completion_tokens, logprobs, top_logprobs, passthrough}
      openai.pivot_judge.{model, temperature, max_completion_tokens}
    """
    raw = _read_yaml(path)

    provider = str(_require(raw, "provider", "models"))
    if provider != "openai":
        raise ConfigError(f"Only provider=openai is supported in V1, got '{provider}'")

    openai_raw = _as_dict(_require(raw, "openai", "models"), "models.openai")
    api_key_env = str(openai_raw.get("api_key_env", "OPENAI_API_KEY"))

    response_raw = _as_dict(_require(openai_raw, "response", "models.openai"), "models.openai.response")
    passthrough = response_raw.get("passthrough", {})
    if passthrough is None:
        passthrough = {}
    if not isinstance(passthrough, dict):
        raise ConfigError("models.openai.response.passthrough must be a dict/map")

    temperature = response_raw.get("temperature")
    temperature = float(temperature) if temperature is not None else None
    response = OpenAIResponseConfig(
        model=str(_require(response_raw, "model", "models.openai.response")),
        temperature=temperature,
        max_completion_tokens=int(_require(response_raw, "max_completion_tokens", "models.openai.response")),
        logprobs=bool(_require(response_raw, "logprobs", "models.openai.response")),
        top_logprobs=int(_require(response_raw, "top_logprobs", "models.openai.response")),
        passthrough=dict(passthrough),
    )

    judge_raw = _as_dict(_require(openai_raw, "pivot_judge", "models.openai"), "models.openai.pivot_judge")
    temperature = judge_raw.get("temperature")
    temperature = float(temperature) if temperature is not None else None
    pivot_judge = OpenAIPivotJudgeConfig(
        model=str(_require(judge_raw, "model", "models.openai.pivot_judge")),
        temperature=temperature,
        max_completion_tokens=int(_require(judge_raw, "max_completion_tokens", "models.openai.pivot_judge")),
    )

    openai = OpenAIConfig(
        api_key_env=api_key_env,
        response=response,
        pivot_judge=pivot_judge,
    )

    return ModelsConfig(provider="openai", openai=openai)