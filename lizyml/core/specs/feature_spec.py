"""FeatureSpec: feature configuration derived from LizyMLConfig."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureSpec:
    """Normalized feature configuration passed downstream."""

    exclude: tuple[str, ...] = field(default_factory=tuple)
    auto_categorical: bool = True
    categorical: tuple[str, ...] = field(default_factory=tuple)
