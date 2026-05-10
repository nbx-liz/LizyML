"""FitState — frozen snapshot of Model state consumed by Mixin methods (#112, H-0074).

Created by ``Model._get_fit_state()`` after ``fit()`` / ``tune()`` / ``load()``.
Mixin methods (``ModelPlotsMixin``, ``ModelTablesMixin``, ``ModelPersistenceMixin``)
will increasingly receive a ``FitState`` instance instead of reading
``self._*`` attributes directly. Phase 1 (this introduction PR) only adds
the dataclass and the factory; Mixin signatures remain backward-compatible.
Phase 2 will migrate Mixin methods to ``state: FitState`` and remove the
``self._*`` access path.

The class is frozen and contains only references — it is cheap to build
once per public-API call. It must NOT capture transient training state
(e.g. per-fold buffers); only the post-fit handles required by diagnostic
methods belong here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    from lizyml.config.schema import LizyMLConfig
    from lizyml.core.types.fit_result import FitResult
    from lizyml.core.types.tuning_result import TuningResult
    from lizyml.estimators.provider import EstimatorProvider
    from lizyml.training.refit_trainer import RefitResult


@dataclass(frozen=True)
class FitState:
    """Frozen snapshot of post-fit Model state for Mixin consumption.

    Attributes:
        cfg: The active :class:`LizyMLConfig` (always present).
        fit_result: Output of :meth:`Model.fit`. Required for any
            diagnostic / plot / table method.
        refit_result: Output of refit on full data. ``None`` when the
            user disabled refit or the model was loaded without it.
        tuning_result: Output of :meth:`Model.tune`. ``None`` when tune
            was not called.
        provider: The estimator provider used for the current model.
            Required for SHAP, params summary, and codegen export.
        metrics: Pre-computed metrics dict (``{"raw": {...}, "calibrated":
            {...}}``). Populated by :meth:`Evaluator.evaluate`.
        y: Training target (transient — absent after :meth:`Model.load` if
            the artifact does not contain ``analysis_context``).
        X: Training features (transient — same caveat as ``y``).
        run_dir: Active run directory for log/artifact output. ``None``
            until ``fit`` / ``tune`` allocate one.
        output_dir: User-configured root for run directories. ``None``
            when neither config nor constructor specified one.
    """

    cfg: LizyMLConfig
    fit_result: FitResult
    refit_result: RefitResult | None
    tuning_result: TuningResult | None
    provider: EstimatorProvider
    metrics: dict[str, Any] | None
    y: pd.Series | None
    X: pd.DataFrame | None
    run_dir: Path | None
    output_dir: str | Path | None
