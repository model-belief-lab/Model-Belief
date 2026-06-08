from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class BeliefItem:
    alt_id: str
    prob: float


@dataclass(frozen=True)
class Belief:
    items: List[BeliefItem]
    total_mass: float
    missing_mass: float
    pivot_index: int
    debug: Dict[str, object]