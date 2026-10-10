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

from lizyml.core.exceptions import ErrorCode, LizyMLError
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


_SAMPLE_WEIGHT_RULES = ("balanced", "none")


def _checked_applied_sample_weight(record: Any, path: str | Path) -> str:
    """Refuse a weight-rule record no fit could have written (H-0120).

    A fit records ``"balanced"`` or ``"none"``; any other value -- ``null``,
    ``true``, another spelling -- would hand the generated ``train.py`` a rule
    nobody measured.
    """
    if not isinstance(record, str) or record not in _SAMPLE_WEIGHT_RULES:
        raise LizyMLError(
            code=ErrorCode.DESERIALIZATION_FAILED,
            user_message=(
                f"Stored applied_sample_weight must be one of "
                f"{list(_SAMPLE_WEIGHT_RULES)}, got {record!r}."
            ),
            context={"path": str(path), "type": type(record).__name__},
        )
    return record


def _checked_declared_categories(record: Any, path: str | Path) -> dict[str, list[Any]]:
    """Refuse a declared-categories record no export could have written (H-0120).

    ``Model.export`` writes an object mapping column names to lists of plain
    JSON scalars (str, int, float, bool); anything else would hand the
    generated ``train.py`` categories nobody recorded.
    """

    def refuse(reason: str, context: dict[str, Any]) -> LizyMLError:
        return LizyMLError(
            code=ErrorCode.DESERIALIZATION_FAILED,
            user_message=f"Stored declared_categories {reason}.",
            context={"path": str(path), **context},
        )

    if not isinstance(record, dict):
        raise refuse("must be an object", {"type": type(record).__name__})
    for col, cats in record.items():
        if not isinstance(cats, list):
            raise refuse(
                f"holds a {type(cats).__name__} for {col!r}, not a list",
                {"key": col, "type": type(cats).__name__},
            )
        for value in cats:
            if type(value) not in (str, int, float, bool):
                raise refuse(
                    f"holds {value!r} for {col!r}, not a str, int, float or bool",
                    {"key": col, "type": type(value).__name__},
                )
    return {col: list(cats) for col, cats in record.items()}


def _checked_applied_training_params(record: Any, path: str | Path) -> dict[str, Any]:
    """Refuse a record no fit could have written (H-0109).

    A fit records its overlay through ``applied_training_overlay``: the two
    training dimensions, converted as training converts them -- the patience an
    ``int``, the ratio a ``float`` in ``(0, 1)``, because every inner-validation
    strategy requires that, so no fit can have applied another. Anything else
    would fail later and elsewhere: ``float()`` inside ``params_table``, or an
    impossible ratio handed to the generated ``train.py``. The patience gets no
    range: the reports read it from the adapter, and training converts it with
    ``int()`` without one.

    Every refusal's context carries the path and the type of the offending
    value, and the key when there is one.
    """
    import math

    from lizyml.core._tuning_validation import TRAINING_DIMENSION_NAMES

    def refuse(reason: str, context: dict[str, Any]) -> LizyMLError:
        return LizyMLError(
            code=ErrorCode.DESERIALIZATION_FAILED,
            user_message=f"Stored applied_training_params {reason}.",
            context={"path": str(path), **context},
        )

    if not isinstance(record, dict):
        raise refuse("must be an object", {"type": type(record).__name__})
    for name, value in record.items():
        context = {"key": name, "type": type(value).__name__}
        if name not in TRAINING_DIMENSION_NAMES:
            raise refuse(
                f"names {name!r}, which is not a training dimension",
                {**context, "accepted": sorted(TRAINING_DIMENSION_NAMES)},
            )
        expected = int if name == "early_stopping_rounds" else float
        # ``bool`` is an ``int`` subclass, and JSON integers are unbounded:
        # ``math.isfinite(10**400)`` raises ``OverflowError``, so only floats
        # are tested for finiteness (code review round 1).
        if (
            isinstance(value, bool)
            or not isinstance(value, expected)
            or (isinstance(value, float) and not math.isfinite(value))
        ):
            raise refuse(
                f"holds {value!r} for {name!r}, which is not a finite "
                f"{expected.__name__}",
                context,
            )
        if name == "validation_ratio" and not 0.0 < value < 1.0:
            raise refuse(
                f"holds validation_ratio={value!r}, outside (0, 1)",
                {**context, "value": value},
            )
    return dict(record)


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
            tuning_fixed_params=state.tuning_fixed_params,
            applied_training_params=state.applied_training_params,
            applied_sample_weight=state.applied_sample_weight,
            declared_categories=state.declared_categories,
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
        from lizyml.core._model_factories import (
            check_param_names,
            get_outer_n_splits,
            tuned_validation_ratio,
        )

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
        tuned_ratio = tuned_validation_ratio(state.applied_training_params)
        effective_ratio = es.validation_ratio if tuned_ratio is None else tuned_ratio

        # H-0120: what the refit trained on, so the generated train.py
        # reproduces it -- the inner split rebuilt by the function the fit used
        # from the inputs it used, the weight rule the fit recorded, and the
        # features the fit's input declared `category` (amendment 4).
        from lizyml.core._codegen_inputs import (
            derived_sample_weight,
            exported_inner_valid,
        )

        inner_valid = exported_inner_valid(cfg, state.applied_training_params)
        if state.applied_sample_weight is not None:
            sample_weight = (
                "balanced" if state.applied_sample_weight == "balanced" else None
            )
        else:
            # Unknown (an artifact written before the record): derived from the
            # config and the current tuning result, outside the promise.
            sample_weight = derived_sample_weight(
                cfg,
                state.provider,
                state.tuning_result.best_smart_params
                if state.tuning_result is not None
                else None,
            )
        # Unknown (an artifact without the record): nothing is restored, so a
        # CSV retrain of such a model is outside the promise; parquet keeps the
        # dtype itself.
        declared = state.declared_categories or {}
        calibration_method: str | None = None
        # Use outer CV n_splits for OOF calibration (H-0058: reuses outer splits)
        calibration_n_splits = get_outer_n_splits(cfg)
        calibration_params: dict[str, Any] = {}
        if cfg.calibration is not None:
            from lizyml.core._model_factories import prepare_calibration_params

            calibration_method = cfg.calibration.method
            # The same preparation the fit applied (H-0100), so the generated
            # train.py rebuilds the calibrator with those settings (H-0059).
            calibration_params = prepare_calibration_params(
                cfg.calibration, seed=cfg.training.seed
            )

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
            # Both of these describe the run the generated project must
            # reproduce, so both come from what the fit applied -- never from
            # the config plus the model's *current* tuning result. Reading the
            # config alone generated a project training a different model after
            # a tune (decision 12); recomputing from the current tuning result
            # generated one training a different model after `fit -> tune`,
            # because `tune()` replaces that result and leaves the fitted
            # adapters alone (decision 13, review round 16 -- a defect decision
            # 12's own fix introduced).
            #
            # The patience is the trained adapter's, through the provider. The
            # ratio is the retained overlay's, because the adapter does not
            # record it; the artifact records the overlay and `load()` restores
            # it (H-0109). Only an artifact written before that record existed
            # leaves it unknown, and then the configured ratio is used -- the
            # bound stated on `FitState.applied_training_params`.
            early_stopping_rounds=export.early_stopping_rounds,
            validation_ratio=effective_ratio or 0.0,
            seed=cfg.training.seed,
            calibration_method=calibration_method,
            calibration_n_splits=calibration_n_splits,
            calibration_params=calibration_params,
            model_adapter=adapter,
            pipeline_state=refit_result.pipeline_state,
            calibrator=calibrator,
            feval_metrics=export.feval_metadata,
            target_classes=target_classes,
            split=_build_split_metadata(cfg),
            inner_valid=inner_valid,
            sample_weight=sample_weight,
            declared_categories=declared,
            categorical_rule={
                "explicit": list(cfg.features.categorical),
                "auto": cfg.features.auto_categorical,
            },
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
        # Pre-H-0102 artifacts used replacement for nonempty user spaces.
        # Do not reinterpret their tuning policy when restoring for re-fit.
        if config.get("tuning") is not None:
            optuna = config["tuning"].get("optuna", {})
            if "space_mode" not in optuna:
                optuna["space_mode"] = "replace" if optuna.get("space") else "merge"
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

            if "fixed_params" in tuning_meta:
                fixed_params = tuning_meta["fixed_params"]
                if not isinstance(fixed_params, dict):
                    raise LizyMLError(
                        code=ErrorCode.DESERIALIZATION_FAILED,
                        user_message="Stored tuning fixed_params must be an object.",
                        context={"path": str(path)},
                    )
                instance._tuning_fixed_params = deepcopy(fixed_params)
            instance._tuning_result = TuningResult(
                best_model_params=tuning_meta["best_model_params"],
                best_smart_params=tuning_meta["best_smart_params"],
                best_training_params=tuning_meta["best_training_params"],
                best_score=tuning_meta["best_score"],
                trials=[],
                metric_name=tuning_meta["metric_name"],
                direction=tuning_meta["direction"],
            )
        # The overlay the fit that produced this artifact applied (H-0109).
        # Absent from artifacts written before the record existed: unknown,
        # which is not the same as "applied none", and must stay unknown so a
        # re-export does not write a record nobody measured.
        if "applied_training_params" in metadata:
            instance._applied_training_params = _checked_applied_training_params(
                metadata["applied_training_params"], path
            )
        else:
            instance._applied_training_params = None
        # The row-weight rule that fit applied (H-0120 amendment 1); absent
        # from older artifacts, which stay unknown for the same reason.
        if "applied_sample_weight" in metadata:
            instance._applied_sample_weight = _checked_applied_sample_weight(
                metadata["applied_sample_weight"], path
            )
        else:
            instance._applied_sample_weight = None
        # The features the fit's input declared `category` (amendment 4).
        if "declared_categories" in metadata:
            instance._declared_categories = _checked_declared_categories(
                metadata["declared_categories"], path
            )
        else:
            instance._declared_categories = None
        if analysis_context is not None:
            instance._y = analysis_context.y_true
            instance._X = analysis_context.X_for_explain
        _log.info("event='load.done' path=%s run_id=%s", path, metadata.get("run_id"))
        return instance
