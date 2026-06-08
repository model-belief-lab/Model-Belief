from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Set, Tuple

from .models import AlternativeConfig, DerivedTokenMaps, TokenPolicyConfig


class ConfigDerivationError(ValueError):
    pass


def _normalize_token(token: str, *, case_sensitive: bool) -> str:
    return token if case_sensitive else token.lower()


def _expand_leading_space_variants(tokens: Iterable[str]) -> Set[str]:
    expanded: Set[str] = set(tokens)
    for t in list(expanded):
        if not t.startswith(" "):
            expanded.add(" " + t)
    return expanded


def derive_token_maps(
    alternatives: List[AlternativeConfig],
    token_policy: TokenPolicyConfig,
) -> DerivedTokenMaps:
    """
    Derive:
      - alt_to_tokens (normalized + optional leading-space variants)
      - token_universe
      - token_to_alt (must be one-to-one; raise if token overlaps across alts)
    """
    case_sensitive = token_policy.case_sensitive
    allow_space = token_policy.allow_leading_space_variant

    alt_to_tokens: Dict[str, Set[str]] = {}
    all_tokens: Set[str] = set()

    for alt in alternatives:
        if not alt.target_tokens:
            raise ConfigDerivationError(f"Alternative '{alt.id}' has empty target_tokens.")

        normed = {_normalize_token(t, case_sensitive=case_sensitive) for t in alt.target_tokens}
        if allow_space:
            normed = _expand_leading_space_variants(normed)

        alt_to_tokens[alt.id] = normed
        all_tokens |= normed

    # token_to_alt must be 1-1
    token_to_alt: Dict[str, str] = {}
    overlaps: List[Tuple[str, str, str]] = []  # (token, alt_existing, alt_new)

    for alt_id, toks in alt_to_tokens.items():
        for t in toks:
            if t in token_to_alt and token_to_alt[t] != alt_id:
                overlaps.append((t, token_to_alt[t], alt_id))
            else:
                token_to_alt[t] = alt_id

    if overlaps:
        # Show a few overlaps to help debugging
        preview = "; ".join([f"'{tok}' in {a1} and {a2}" for tok, a1, a2 in overlaps[:10]])
        raise ConfigDerivationError(
            "Target tokens overlap across alternatives. "
            f"Examples: {preview}"
        )

    normalized = not case_sensitive
    return DerivedTokenMaps(
        alt_to_tokens=alt_to_tokens,
        token_to_alt=token_to_alt,
        token_universe=set(all_tokens),
        normalized=normalized,
    )