"""SearchDim — optuna-independent hyperparameter dimension types (Foundation layer).

These types are pure data (frozen dataclasses) with no external dependencies beyond
the standard library.  They live in Foundation (L0) so that both Leaf-layer modules
(e.g. ``estimators/provider.py``) and Composition-layer modules
(e.g. ``tuning/search_space.py``) can reference them without creating upward
dependencies.

Moved from ``tuning/search_space.py`` as part of H-0054.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

DimCategory = Literal["model", "smart", "training"]


@dataclass(frozen=True)
class FloatDim:
    """A continuous float hyperparameter dimension."""

    name: str
    low: float
    high: float
    log: bool = False
    category: DimCategory = "model"


@dataclass(frozen=True)
class IntDim:
    """An integer hyperparameter dimension."""

    name: str
    low: int
    high: int
    log: bool = False
    category: DimCategory = "model"


@dataclass(frozen=True)
class CategoricalDim:
    """A categorical hyperparameter dimension."""

    name: str
    choices: tuple[Any, ...]
    category: DimCategory = "model"


SearchDim = FloatDim | IntDim | CategoricalDim
