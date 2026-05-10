"""Regression tests for issue #95.

`EarlyStoppingConfig.model_dump()` always emits both ``inner_valid`` and
``validation_ratio`` (the latter from its non-None default 0.1). The
round-trip allowance previously only accepted ``HoldoutInnerValidConfig``,
so models persisted with ``group_holdout`` or ``time_holdout`` could not be
re-loaded via ``LizyMLConfig.model_validate(model_dump())`` (i.e.
``Model.load()``).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lizyml.config.schema import EarlyStoppingConfig


class TestInnerValidRoundtrip:
    """Round-trip ``model_dump()`` → ``model_validate()`` for every
    ``InnerValidConfig`` discriminant."""

    @pytest.mark.parametrize(
        "inner_valid",
        [
            {"method": "holdout", "ratio": 0.1, "stratify": False, "random_state": 42},
            {"method": "group_holdout", "ratio": 0.1, "random_state": 42},
            {"method": "time_holdout", "ratio": 0.1},
        ],
        ids=["holdout", "group_holdout", "time_holdout"],
    )
    def test_roundtrip_preserves_inner_valid(
        self, inner_valid: dict[str, object]
    ) -> None:
        cfg = EarlyStoppingConfig(
            enabled=True,
            rounds=50,
            inner_valid=inner_valid,  # type: ignore[arg-type]
        )
        dumped = cfg.model_dump()

        # Sanity: dump must include both fields (this is what triggered the bug).
        assert "inner_valid" in dumped
        assert "validation_ratio" in dumped

        restored = EarlyStoppingConfig.model_validate(dumped)
        assert restored.inner_valid is not None
        assert restored.inner_valid.method == inner_valid["method"]
        assert restored.inner_valid.ratio == inner_valid["ratio"]

    @pytest.mark.parametrize(
        "method",
        ["holdout", "group_holdout", "time_holdout"],
    )
    def test_roundtrip_with_non_default_ratio(self, method: str) -> None:
        """Non-default ratio still round-trips (ratio is propagated to
        ``validation_ratio`` when only ``inner_valid`` is set)."""
        cfg = EarlyStoppingConfig(
            enabled=True,
            rounds=50,
            inner_valid={"method": method, "ratio": 0.25},  # type: ignore[arg-type]
        )
        dumped = cfg.model_dump()
        restored = EarlyStoppingConfig.model_validate(dumped)
        assert restored.inner_valid is not None
        assert restored.inner_valid.ratio == 0.25


class TestInnerValidConflictGuard:
    """The 'specify only one' guard must still fire on a *real* conflict —
    inconsistent ``inner_valid.ratio`` and ``validation_ratio``."""

    @pytest.mark.parametrize(
        "method",
        ["holdout", "group_holdout", "time_holdout"],
    )
    def test_inconsistent_ratio_still_rejected(self, method: str) -> None:
        with pytest.raises(ValidationError) as exc_info:
            EarlyStoppingConfig.model_validate(
                {
                    "enabled": True,
                    "rounds": 50,
                    "inner_valid": {"method": method, "ratio": 0.1},
                    "validation_ratio": 0.25,
                }
            )
        assert "Specify either 'validation_ratio' or 'inner_valid'" in str(
            exc_info.value
        )


class TestExplicitFlagTracking:
    """``_inner_valid_explicit`` PrivateAttr drives the factory's
    auto-resolve path (`_model_factories.py:253`).  H-0069 must
    preserve the existing semantics:

    - Legacy ``validation_ratio`` only → ``False`` (auto-resolve)
    - Explicit ``inner_valid`` only → ``True`` (use as-is)
    - Round-trip (both keys) → ``False`` (auto-resolve, matching
      pre-H-0069 behaviour for round-tripped artifacts)
    """

    def test_legacy_validation_ratio_keeps_auto_resolve(self) -> None:
        with pytest.warns(DeprecationWarning, match="validation_ratio"):
            cfg = EarlyStoppingConfig.model_validate(
                {"enabled": True, "rounds": 50, "validation_ratio": 0.2}
            )
        assert cfg.inner_valid is not None
        assert cfg.inner_valid.method == "holdout"
        assert cfg.inner_valid.ratio == 0.2
        assert cfg._inner_valid_explicit is False

    def test_explicit_inner_valid_disables_auto_resolve(self) -> None:
        cfg = EarlyStoppingConfig.model_validate(
            {
                "enabled": True,
                "rounds": 50,
                "inner_valid": {"method": "group_holdout", "ratio": 0.2},
            }
        )
        assert cfg._inner_valid_explicit is True
        assert cfg.validation_ratio == 0.2

    def test_default_construction_keeps_auto_resolve(self) -> None:
        cfg = EarlyStoppingConfig()
        assert cfg.inner_valid is not None
        assert cfg.inner_valid.ratio == 0.1
        assert cfg._inner_valid_explicit is False

    def test_roundtrip_preserves_auto_resolve(self) -> None:
        """A dump always carries both ``inner_valid`` and ``validation_ratio``
        — re-loading must mirror the legacy behaviour where the pair triggers
        auto-resolve, not explicit."""
        original = EarlyStoppingConfig.model_validate(
            {
                "enabled": True,
                "rounds": 50,
                "inner_valid": {"method": "group_holdout", "ratio": 0.2},
            }
        )
        restored = EarlyStoppingConfig.model_validate(original.model_dump())
        assert restored._inner_valid_explicit is False


class TestComputedFieldContract:
    """``validation_ratio`` is now a computed field — it must always
    mirror ``inner_valid.ratio`` and be present in ``model_dump()``."""

    @pytest.mark.parametrize(
        ("method", "ratio"),
        [
            ("holdout", 0.1),
            ("holdout", 0.3),
            ("group_holdout", 0.2),
            ("time_holdout", 0.4),
        ],
    )
    def test_validation_ratio_mirrors_inner_valid(
        self, method: str, ratio: float
    ) -> None:
        cfg = EarlyStoppingConfig(
            inner_valid={"method": method, "ratio": ratio}  # type: ignore[arg-type]
        )
        assert cfg.validation_ratio == ratio
        dumped = cfg.model_dump()
        assert dumped["validation_ratio"] == ratio
        assert dumped["inner_valid"]["ratio"] == ratio

    def test_validation_ratio_is_read_only(self) -> None:
        """Direct assignment to the computed field must fail —
        ``inner_valid.ratio`` is the SSOT."""
        cfg = EarlyStoppingConfig()
        with pytest.raises((AttributeError, ValueError)):
            cfg.validation_ratio = 0.5  # type: ignore[misc]
