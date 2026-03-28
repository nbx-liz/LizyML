"""ModelPersistenceMixin — export/load methods extracted from Model facade."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.logging import generate_run_id, get_logger

if TYPE_CHECKING:
    import pandas as pd

    from lizyml.config.schema import LizyMLConfig
    from lizyml.core.types.fit_result import FitResult
    from lizyml.estimators.provider import EstimatorProvider
    from lizyml.training.refit_trainer import RefitResult

_log = get_logger("model")


class ModelPersistenceMixin:
    """Mixin providing export/load methods for :class:`Model`."""

    # Attributes provided by Model — declared for type checking only.
    if TYPE_CHECKING:
        _cfg: LizyMLConfig
        _fit_result: FitResult | None
        _refit_result: RefitResult | None
        _metrics: dict[str, Any] | None
        _y: pd.Series | None
        _X: pd.DataFrame | None
        _run_dir: Path | None
        _output_dir: str | Path | None
        _provider: EstimatorProvider | None

        def _require_fit(self) -> FitResult: ...

        def _require_refit(self) -> RefitResult: ...

    def export(self, path: str | Path | None = None) -> Path:
        """Export Model artifacts to a directory.

        Saves ``fit_result.pkl``, ``refit_model.pkl``, ``metadata.json``,
        and ``analysis_context.pkl`` under *path*.  The saved model can be
        restored with :meth:`load`, including diagnostic API support.

        Path resolution (first match wins):

        1. Explicit *path* argument.
        2. ``{run_dir}/export`` when a run directory exists from ``fit``/``tune``.
        3. New run directory under ``output_dir`` if configured.
        4. Error — no destination available.

        Args:
            path: Output directory (created if absent).  Optional when
                ``output_dir`` is configured via Config or constructor.

        Returns:
            Resolved export directory path.

        Raises:
            LizyMLError with MODEL_NOT_FIT when called before ``fit``.
            LizyMLError with SERIALIZATION_FAILED on I/O errors or when
                no path can be resolved.

        Warning:
            The ``.pkl`` files use joblib/pickle.  Only load artifacts from
            trusted sources.
        """
        fit_result = self._require_fit()
        refit_result = self._require_refit()

        resolved_path = self._resolve_export_path(path)

        from lizyml.persistence.exporter import AnalysisContext
        from lizyml.persistence.exporter import export as _export

        ctx: AnalysisContext | None = None
        if self._y is not None and self._X is not None:
            ctx = AnalysisContext(y_true=self._y, X_for_explain=self._X)

        _export(
            path=resolved_path,
            fit_result=fit_result,
            refit_result=refit_result,
            config=self._cfg.model_dump(),
            task=self._cfg.task,
            analysis_context=ctx,
        )
        _log.info("event='export.done' path=%s", resolved_path)
        return resolved_path

    def export_code(self, path: str | Path) -> Path:
        """Generate LizyML-independent training and prediction code.

        Creates ``train.py``, ``predict.py``, ``config.json``,
        ``requirements.txt``, and ``artifacts/`` under *path*.

        Args:
            path: Output directory (created if absent).

        Returns:
            Resolved output directory path.

        Raises:
            LizyMLError with ``MODEL_NOT_FIT`` when called before ``fit``.
        """
        fit_result = self._require_fit()
        refit_result = self._require_refit()

        from lizyml.codegen.generator import generate_code
        from lizyml.estimators.lgbm.adapter import LGBMAdapter

        adapter = refit_result.model
        if not isinstance(adapter, LGBMAdapter):
            raise LizyMLError(
                ErrorCode.UNSUPPORTED_TASK,
                user_message=("export_code() currently supports LGBMAdapter only."),
            )

        # Extract LightGBM params from the adapter.
        # TODO(H-0059): expose via EstimatorProvider protocol in a future PR.
        lgbm_params, num_boost_round, _ = adapter._build_params()

        cfg = self._cfg
        es = cfg.training.early_stopping
        calibration_method: str | None = None
        # Use outer CV n_splits for OOF calibration (H-0058: reuses outer splits)
        from lizyml.config.schema import BlockedGroupKFoldConfig

        if isinstance(cfg.split, BlockedGroupKFoldConfig):
            calibration_n_splits = cfg.split.groups.n_splits
        else:
            calibration_n_splits = cfg.split.n_splits
        if cfg.calibration is not None:
            calibration_method = cfg.calibration.method

        # Extract c_final calibrator from CalibrationResult
        calibrator = None
        cal_result = fit_result.calibrator
        if cal_result is not None and hasattr(cal_result, "c_final"):
            calibrator = cal_result.c_final

        # Build run_meta dict from FitResult
        meta = fit_result.run_meta
        run_meta_dict: dict[str, Any] = {
            "lizyml_version": meta.lizyml_version,
            "run_id": meta.run_id,
            "timestamp": meta.timestamp,
            "config_normalized": meta.config_normalized,
        }

        result = generate_code(
            output_dir=path,
            run_meta=run_meta_dict,
            feature_names=refit_result.feature_names,
            categorical_features=refit_result.categorical_features,
            lgbm_params=lgbm_params,
            num_boost_round=num_boost_round,
            early_stopping_rounds=(es.rounds if es.enabled else None),
            validation_ratio=es.validation_ratio or 0.0,
            seed=cfg.training.seed,
            calibration_method=calibration_method,
            calibration_n_splits=calibration_n_splits,
            model_adapter=adapter,
            pipeline_state=refit_result.pipeline_state,
            calibrator=calibrator,
        )
        _log.info("event='export_code.done' path=%s", result)
        return result

    def _resolve_export_path(self, path: str | Path | None) -> Path:
        """Resolve the export destination directory."""
        if path is not None:
            return Path(path)
        if self._run_dir is not None:
            return Path(self._run_dir) / "export"
        if self._output_dir is not None:
            from lizyml.core.logging import setup_output_dir

            export_run_id = generate_run_id()
            self._run_dir = setup_output_dir(self._output_dir, export_run_id)
            return Path(self._run_dir) / "export"
        raise LizyMLError(
            ErrorCode.SERIALIZATION_FAILED,
            user_message=(
                "No export path provided and no output_dir configured. "
                "Pass an explicit path or set output_dir in Config / constructor."
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> Any:
        """Restore a Model from a directory created by :meth:`export`.

        Args:
            path: Directory containing ``metadata.json``, ``fit_result.pkl``,
                and ``refit_model.pkl``.

        Returns:
            A :class:`Model` instance ready for ``predict`` and ``evaluate``.

        Raises:
            LizyMLError with DESERIALIZATION_FAILED on validation or I/O errors.

        Warning:
            Only load from trusted sources — joblib uses pickle internally.
        """
        from lizyml.persistence.loader import load as _load

        fit_result, refit_result, metadata, analysis_context = _load(path)
        config = metadata["config"]
        instance = cls(config)  # type: ignore[call-arg]  # cls is Model at runtime
        instance._fit_result = fit_result
        instance._refit_result = refit_result
        instance._metrics = fit_result.metrics
        # Restore provider for params_table() etc. (H-0054)
        from lizyml.core._model_factories import get_provider

        instance._provider = get_provider(instance._cfg.model)
        if analysis_context is not None:
            instance._y = analysis_context.y_true
            instance._X = analysis_context.X_for_explain
        _log.info("event='load.done' path=%s run_id=%s", path, metadata.get("run_id"))
        return instance
