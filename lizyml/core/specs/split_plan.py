"""SplitPlan — integrates outer / inner_valid / calibration splitters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lizyml.core.specs.calibration_spec import CalibrationSpec
from lizyml.core.specs.split_spec import SplitSpec
from lizyml.core.specs.training_spec import TrainingSpec

if TYPE_CHECKING:
    from lizyml.splitters.base import BaseSplitter


@dataclass
class SplitPlan:
    """Resolved splitter instances for every split tier.

    Attributes:
        outer: Primary CV splitter for the outer loop.
        inner: Splitter for inner validation (early stopping); ``None`` when
            early stopping is disabled or uses a static holdout ratio.
        calibration: Splitter for calibration cross-fit; ``None`` when
            calibration is not configured.
    """

    outer: BaseSplitter
    inner: BaseSplitter | None
    calibration: BaseSplitter | None

    @classmethod
    def create(
        cls,
        split_spec: SplitSpec,
        training_spec: TrainingSpec,
        calibration_spec: CalibrationSpec | None = None,
    ) -> SplitPlan:
        """Build a SplitPlan from the three spec objects.

        Args:
            split_spec: Outer CV configuration.
            training_spec: Training configuration (early stopping / inner valid).
            calibration_spec: Optional calibration configuration.

        Returns:
            A fully constructed ``SplitPlan``.
        """
        from lizyml.splitters import _build_splitter

        outer = _build_splitter(split_spec)

        inner: BaseSplitter | None = None
        iv = training_spec.early_stopping.inner_valid
        if iv is not None:
            inner = _build_splitter_for_inner(iv)

        calibration: BaseSplitter | None = None
        if calibration_spec is not None:
            calibration = _build_calibration_splitter(calibration_spec)

        return cls(outer=outer, inner=inner, calibration=calibration)


def _build_splitter_for_inner(iv: Any) -> BaseSplitter:
    """Build the inner validation splitter from an InnerValidSpec."""
    from lizyml.splitters.holdout import HoldoutSplitter

    return HoldoutSplitter(
        ratio=iv.ratio,
        random_state=iv.random_state,
    )


def _build_calibration_splitter(cal: CalibrationSpec) -> BaseSplitter:
    """Build the calibration CV splitter from a CalibrationSpec."""
    from lizyml.splitters.kfold import KFoldSplitter

    return KFoldSplitter(
        n_splits=cal.n_splits,
        shuffle=True,
        random_state=42,
    )
