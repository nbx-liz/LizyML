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
    """A continuous float hyperparameter dimension.

    ``min_allowed`` / ``max_allowed`` describe the *parameter-meaningful*
    bounds (e.g. ``learning_rate`` cannot exceed 1.0).  When set, boundary
    expansion (``expand_dims``) clamps the new range to these limits.
    See H-0078 for context.
    """

    name: str
    low: float
    high: float
    log: bool = False
    category: DimCategory = "model"
    min_allowed: float | None = None
    max_allowed: float | None = None


@dataclass(frozen=True)
class IntDim:
    """An integer hyperparameter dimension.

    See :class:`FloatDim` for ``min_allowed`` / ``max_allowed`` semantics.
    """

    name: str
    low: int
    high: int
    log: bool = False
    category: DimCategory = "model"
    min_allowed: int | None = None
    max_allowed: int | None = None


@dataclass(frozen=True)
class CategoricalDim:
    """A categorical hyperparameter dimension."""

    name: str
    choices: tuple[Any, ...]
    category: DimCategory = "model"


SearchDim = FloatDim | IntDim | CategoricalDim
