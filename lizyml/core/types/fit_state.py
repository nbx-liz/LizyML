"""FitState / TuningState — frozen snapshots of Model state for Mixin methods (#112).

Created by ``Model._get_fit_state()`` / ``Model._get_tuning_state()`` and consumed by
``ModelPlotsMixin`` / ``ModelTablesMixin`` / ``ModelPersistenceMixin``. After H-0077
(Phase 2) Mixin methods read state exclusively from these snapshots — direct
``self._*`` access is forbidden inside Mixin bodies.

Two state types are distinguished by lifecycle:

* :class:`FitState` — post-``fit()`` / ``tune()`` / ``load()`` snapshot. Required
  for any diagnostic API that depends on ``fit_result`` (plots, tables, export).
* :class:`TuningState` — post-``tune()`` snapshot, valid even before ``fit()`` is
  called. Required for ``tuning_plot`` / ``tuning_table`` / ``boundary_table``.

Both classes are frozen and contain only references — they are cheap to build
once per public-API call. They must NOT capture transient training state
(e.g. per-fold buffers); only the handles required by diagnostic / export
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


@dataclass(frozen=True)
class TuningState:
    """Frozen snapshot of post-``tune()`` Model state for Mixin consumption.

    Used by ``tuning_plot`` / ``tuning_table`` / ``boundary_table`` which must
    work after ``tune()`` even when ``fit()`` has not been called. Keeping this
    distinct from :class:`FitState` preserves the latter's "fit-required"
    invariant and avoids forcing every Mixin method to handle a ``None``
    ``fit_result``.

    Attributes:
        cfg: The active :class:`LizyMLConfig`.
        tuning_result: Output of :meth:`Model.tune`. Always non-``None`` —
            :meth:`Model._get_tuning_state` raises ``MODEL_NOT_FIT`` when
            ``tune()`` has not been called.
    """

    cfg: LizyMLConfig
    tuning_result: TuningResult
