"""Component registries for LizyML.

Each registry maps string identifiers to implementation classes.
All registries support decorator-based registration.
"""

from __future__ import annotations

from typing import Any


class _Registry:
    """Generic string-keyed registry."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._store: dict[str, type[Any]] = {}

    def register(self, key: str) -> Any:
        """Decorator: register a class under *key*."""

        def decorator(cls: type[Any]) -> type[Any]:
            self._store[key] = cls
            return cls

        return decorator

    def get(self, key: str) -> type[Any]:
        """Return the class registered under *key*.

        Raises:
            KeyError: When *key* is not registered.
        """
        if key not in self._store:
            available = list(self._store.keys())
            raise KeyError(
                f"'{key}' is not registered in {self._name}. Available: {available}"
            )
        return self._store[key]

    def keys(self) -> list[str]:
        return list(self._store.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._store


EstimatorRegistry = _Registry("EstimatorRegistry")
SplitterRegistry = _Registry("SplitterRegistry")
MetricRegistry = _Registry("MetricRegistry")
CalibratorRegistry = _Registry("CalibratorRegistry")
