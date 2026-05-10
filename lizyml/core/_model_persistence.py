"""ModelPersistenceMixin — export/load methods extracted from Model facade.

After H-0077 (Phase 2) every method reads state exclusively through
``self._get_fit_state()`` — direct ``self._<private>`` access is
forbidden. Path resolution that mutates ``Model._run_dir`` lives on the
Model facade as ``Model._resolve_export_path``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from lizyml.core.logging import get_logger

if TYPE_CHECKING:
    from lizyml.core.types.fit_state import FitState
    from lizyml.training.refit_trainer import RefitResult

_log = get_logger("model")


class ModelPersistenceMixin:
    """Mixin providing export/load methods for :class:`Model`."""

    # Facade entry points provided by Model — declared for type checking only.
    if TYPE_CHECKING:

        def _get_fit_state(self) -> FitState: ...

        def _require_refit(self) -> RefitResult: ...

        def _resolve_export_path(self, path: str | Path | None) -> Path: ...

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
        state = self._get_fit_state()
        refit_result = self._require_refit()

        resolved_path = self._resolve_export_path(path)

        from lizyml.persistence.exporter import AnalysisContext
        from lizyml.persistence.exporter import export as _export

        ctx: AnalysisContext | None = None
        if state.y is not None and state.X is not None:
            ctx = AnalysisContext(y_true=state.y, X_for_explain=state.X)

        _export(
            path=resolved_path,
            fit_result=state.fit_result,
            refit_result=refit_result,
            config=state.cfg.model_dump(),
            task=state.cfg.task,
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
        state = self._get_fit_state()
        refit_result = self._require_refit()

        from lizyml.codegen.generator import generate_code
        from lizyml.core._model_factories import get_outer_n_splits

        adapter = refit_result.model

        # Codegen-relevant params and feval metadata go through the
        # EstimatorProvider so that this module remains
        # estimator-agnostic (H-0073).
        export = state.provider.build_export_params(adapter)

        cfg = state.cfg
        es = cfg.training.early_stopping
        calibration_method: str | None = None
        # Use outer CV n_splits for OOF calibration (H-0058: reuses outer splits)
        calibration_n_splits = get_outer_n_splits(cfg)
        if cfg.calibration is not None:
            calibration_method = cfg.calibration.method

        # Extract c_final calibrator from CalibrationResult
        calibrator = None
        cal_result = state.fit_result.calibrator
        if cal_result is not None and hasattr(cal_result, "c_final"):
            calibrator = cal_result.c_final

        # Build run_meta dict from FitResult
        meta = state.fit_result.run_meta
        run_meta_dict: dict[str, Any] = {
            "lizyml_version": meta.lizyml_version,
            "run_id": meta.run_id,
            "timestamp": meta.timestamp,
            "config_normalized": meta.config_normalized,
        }

        # H-0070: bake target encoder classes into config so train.py /
        # predict.py can re-encode and decode the original labels.
        target_classes: list[Any] | None = None
        if state.fit_result.target_encoder.needs_encoding:
            target_classes = list(state.fit_result.target_encoder.classes_)

        result = generate_code(
            output_dir=path,
            run_meta=run_meta_dict,
            feature_names=refit_result.feature_names,
            categorical_features=refit_result.categorical_features,
            lgbm_params=export.params,
            num_boost_round=export.num_boost_round,
            early_stopping_rounds=(es.rounds if es.enabled else None),
            validation_ratio=es.validation_ratio or 0.0,
            seed=cfg.training.seed,
            calibration_method=calibration_method,
            calibration_n_splits=calibration_n_splits,
            model_adapter=adapter,
            pipeline_state=refit_result.pipeline_state,
            calibrator=calibrator,
            feval_metrics=export.feval_metadata,
            target_classes=target_classes,
        )
        _log.info("event='export_code.done' path=%s", result)
        return result

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
        # ``load`` is the canonical re-hydration path — direct private-attr
        # writes here are confined to this classmethod and intentionally
        # rebuild the Model body. The Mixin state-isolation guard targets
        # instance methods only.
        instance: Any = cls(config)  # type: ignore[call-arg]  # cls is Model at runtime
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
