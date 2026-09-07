"""ModelPersistenceMixin — export/load methods extracted from Model facade.

After H-0077 (Phase 2) every method reads state exclusively through
``self._get_fit_state()`` — direct ``self._<private>`` access is
forbidden. Path resolution that mutates ``Model._run_dir`` lives on the
Model facade as ``Model._resolve_export_path``.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lizyml.core.logging import get_logger

if TYPE_CHECKING:
    from lizyml.core._model_state import FitState
    from lizyml.training.refit_trainer import RefitResult

_log = get_logger("model")


def _build_split_metadata(cfg: Any) -> dict[str, Any]:
    """Serialize the outer split config so the generated ``train.py`` can
    reproduce the model's CV folds (leakage-safe retrain, #228).

    All method-specific parameters are resolved to plain JSON-serializable
    values (e.g. ``stratify="auto"`` collapsed to a bool, ``random_state``
    fallen back to ``training.seed``) so the template needs no LizyML logic.
    """
    from lizyml.config.schema import (
        BlockedGroupKFoldConfig,
        GroupTimeSeriesConfig,
        KFoldConfig,
        PurgedTimeSeriesConfig,
        StratifiedGroupKFoldConfig,
        StratifiedKFoldConfig,
        TimeSeriesConfig,
    )
    from lizyml.core._model_factories import get_outer_n_splits

    sc = cfg.split
    seed = cfg.training.seed
    block: dict[str, Any] = {
        "method": sc.method,
        "n_splits": get_outer_n_splits(cfg),
        "time_col": cfg.data.time_col,
        "group_col": cfg.data.group_col,
    }
    if isinstance(sc, KFoldConfig):
        # KFoldSplitter uses the config shuffle; StratifiedKFoldSplitter forces
        # shuffle=True (handled below). random_state falls back to training.seed.
        block["shuffle"] = sc.shuffle
        block["random_state"] = sc.random_state if sc.random_state is not None else seed
    elif isinstance(sc, StratifiedKFoldConfig):
        block["shuffle"] = True
        block["random_state"] = sc.random_state if sc.random_state is not None else seed
    elif isinstance(sc, TimeSeriesConfig | GroupTimeSeriesConfig):
        block["gap"] = sc.gap
        block["train_size_max"] = sc.train_size_max
        block["test_size_max"] = sc.test_size_max
    elif isinstance(sc, PurgedTimeSeriesConfig):
        block["purge_gap"] = sc.purge_gap
        block["embargo"] = sc.embargo
        block["train_size_max"] = sc.train_size_max
        block["test_size_max"] = sc.test_size_max
    elif isinstance(sc, StratifiedGroupKFoldConfig):
        block["shuffle"] = sc.shuffle
        block["random_state"] = sc.random_state if sc.random_state is not None else seed
    elif isinstance(sc, BlockedGroupKFoldConfig):
        stratify = sc.groups.stratify
        stratify_bool = (
            cfg.task in ("binary", "multiclass")
            if stratify == "auto"
            else bool(stratify)
        )
        block["blocks"] = {
            "col": sc.blocks.col,
            "cutoffs": list(sc.blocks.cutoffs),
            "mode": sc.blocks.mode,
            "train_window": sc.blocks.train_window,
        }
        block["groups"] = {
            "col": sc.groups.col,
            "n_splits": sc.groups.n_splits,
            "stratify": stratify_bool,
            "shuffle": sc.groups.shuffle,
        }
        block["random_state"] = seed
        block["min_train_rows"] = sc.min_train_rows
        block["min_valid_rows"] = sc.min_valid_rows
    return block


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
            tuning=state.tuning_result,
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
        from lizyml.core._model_factories import check_param_names, get_outer_n_splits

        adapter = refit_result.model

        # Codegen-relevant params and feval metadata go through the
        # EstimatorProvider so that this module remains
        # estimator-agnostic (H-0073).
        export = state.provider.build_export_params(adapter)

        # H-0093: the generated `train.py` hands these straight to `lgb.train`
        # and gets the same silent discard the library gives any unknown name.
        # They come from the fitted adapter, not from the config, so neither
        # gate on the training path sees them -- and `Model.load()` is
        # deliberately permissive, so an artifact written before that gate
        # existed can carry a name LightGBM never honoured right into the
        # exported script.
        check_param_names(
            state.provider,
            (("exported lgbm_params", name) for name in export.params),
            model_name=state.cfg.model.name,
        )

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
            split=_build_split_metadata(cfg),
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
        # Deep-copy the metrics dict so the internal state does not share a
        # mutable object with the ``fit_result`` copy handed to callers
        # (#204 / H-0086 — the same isolation fit()/fit_result enforce).
        instance._metrics = deepcopy(fit_result.metrics)
        # Restore provider for params_table() etc. (H-0054)
        from lizyml.core._model_factories import get_provider

        instance._provider = get_provider(instance._cfg.model)
        # Restore the tuned-param overlay so a re-fit() reproduces the tuned
        # params instead of silently reverting to config defaults (H-0086,
        # #215). Absent for non-tuned / pre-#215 artifacts (stays None).
        tuning_meta = metadata.get("tuning")
        if tuning_meta is not None:
            from lizyml.core.types.tuning_result import TuningResult

            instance._tuning_result = TuningResult(
                best_model_params=tuning_meta["best_model_params"],
                best_smart_params=tuning_meta["best_smart_params"],
                best_training_params=tuning_meta["best_training_params"],
                best_score=tuning_meta["best_score"],
                trials=[],
                metric_name=tuning_meta["metric_name"],
                direction=tuning_meta["direction"],
            )
        if analysis_context is not None:
            instance._y = analysis_context.y_true
            instance._X = analysis_context.X_for_explain
        _log.info("event='load.done' path=%s run_id=%s", path, metadata.get("run_id"))
        return instance
