"""ModelTuningMixin — tune() orchestration extracted from the Model facade (H-0091).

Writer-exempt *orchestrator* mixin. Unlike the read-only *diagnostic* mixins
(``_model_plots`` / ``_model_tables`` / ``_model_persistence``, whose read-only
contract is enforced by ``tests/test_core/test_mixin_state_isolation.py``), this
mixin runs during the mutating ``tune()`` lifecycle and deliberately reads and
writes Model body state (``self._tuning_result`` / ``_study`` / ``_rounds`` /
``_round_number`` / ``_space`` / ``_used_default_space`` ...). It is
intentionally **not** listed in that guard's ``_MIXIN_FILES``; the read-only
H-0077 invariant applies only to the diagnostic mixins. See HISTORY H-0091 (#237).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from lizyml.config.schema import OptunaParamsConfig
from lizyml.core._model_factories import build_splitter, get_provider
from lizyml.core._model_metrics import _DEFAULT_METRICS
from lizyml.core._model_state import TuningState
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.logging import generate_run_id, get_logger
from lizyml.core.train_components import TrainComponents
from lizyml.core.types.artifacts import RunMeta
from lizyml.core.types.tuning_result import (
    BoundaryReport,
    RoundSummary,
    TuneProgressCallback,
    TuningResult,
)
from lizyml.data.fingerprint import compute as fp_compute
from lizyml.evaluation.evaluator import Evaluator
from lizyml.metrics.registry import parse_metric_entry
from lizyml.training.cv_trainer import CVTrainer
from lizyml.tuning.rounds import assemble_round_result
from lizyml.tuning.search_space import (
    attach_bounds,
    detect_boundary,
    expand_dims,
    parse_space,
    split_by_category,
    suggest_params,
)
from lizyml.tuning.tuner import Tuner

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt
    import pandas as pd
    from optuna.storages import BaseStorage

    from lizyml.config.schema import LizyMLConfig
    from lizyml.core.types.artifacts import DataFingerprint
    from lizyml.data.dataframe_builder import DataFrameComponents
    from lizyml.estimators.provider import EstimatorProvider
    from lizyml.splitters.base import BaseSplitter

_log = get_logger("model")


class ModelTuningMixin:
    """Hyperparameter tuning orchestration for :class:`~lizyml.core.model.Model`.

    Writer-exempt orchestrator mixin (H-0091) — see the module docstring for the
    diagnostic-vs-orchestrator mixin category distinction.
    """

    if TYPE_CHECKING:
        # --- Model body state (writer-exempt access during tune lifecycle) ---
        _cfg: LizyMLConfig
        _run_dir: Path | None
        _tuning_result: TuningResult | None
        _y: pd.Series | None
        _X: pd.DataFrame | None
        _provider: EstimatorProvider | None
        _block_values: npt.NDArray[Any] | None
        _study: Any
        _round_number: int
        _rounds: list[RoundSummary]
        _space: list[Any] | None
        _used_default_space: bool

        # --- Facade methods this mixin delegates to (defined on Model) ---
        def _prepare_training_data(
            self, data: pd.DataFrame | None
        ) -> tuple[
            pd.DataFrame, pd.Series, npt.NDArray[Any] | None, DataFrameComponents
        ]: ...

        def _merge_params(
            self, provider: Any, override: dict[str, Any] | None = None
        ) -> tuple[dict[str, Any], dict[str, Any]]: ...

        def _ensure_run_dir(self, run_id: str) -> None: ...

        def _build_train_components(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            *,
            provider: Any,
            model_params: dict[str, Any],
            smart_params: dict[str, Any],
            training_overrides: dict[str, Any] | None = None,
        ) -> TrainComponents: ...

        def _build_run_meta(self, run_id: str) -> RunMeta: ...

    def tune(
        self,
        data: pd.DataFrame | None = None,
        *,
        resume: bool = False,
        n_trials: int | None = None,
        expand_boundary: bool | None = None,
        boundary_threshold: float = 0.05,
        progress_callback: TuneProgressCallback | None = None,
        storage: str | BaseStorage | None = None,
        study_name: str | None = None,
    ) -> TuningResult:
        """Run hyperparameter search with optuna (H-0068: resume + expand,
        H-0072: persistent storage).

        Requires ``tuning`` section in the config.  Best params are stored
        internally and used automatically in the next ``fit()`` call.

        Args:
            data: Training DataFrame.  Overrides any data from construction
                or ``data.path`` in config.
            resume: If True, resume from the previous Study and add trials.
                The TPE sampler reuses knowledge from previous trials.
            n_trials: Number of trials to run.  Defaults to
                ``config.tuning.optuna.params.n_trials`` if None.
            expand_boundary: Whether to auto-expand search space dimensions
                whose best params are near the boundary.  None means True
                for default space, False for user-specified space.
            boundary_threshold: Edge detection threshold; must be in the open
                interval ``(0.0, 0.5)``.
            progress_callback: Optional callback invoked after each trial.
            storage: Optional Optuna storage URL or ``BaseStorage`` instance
                for resumable tuning (H-0072). ``None`` keeps the in-memory
                behavior. When set, ``study_name`` is required.
            study_name: Optional study identifier used together with
                ``storage`` (H-0072). Required when ``storage`` is given.
                Re-using the same ``(storage, study_name)`` pair re-attaches
                to the persisted study via ``load_if_exists=True``.

        Returns:
            :class:`~lizyml.core.types.tuning_result.TuningResult` with
            best params, best score, and full trial history.

        Raises:
            LizyMLError with CONFIG_INVALID when no ``tuning`` config is set.
            LizyMLError with OPTIONAL_DEP_MISSING when optuna is not installed.
            LizyMLError with TUNING_FAILED on study failure or resume without
                prior tune().
        """
        cfg = self._cfg
        self._validate_tune_inputs(resume=resume, boundary_threshold=boundary_threshold)

        optuna_cfg = cfg.tuning.optuna.params  # type: ignore[union-attr]
        actual_n_trials = n_trials if n_trials is not None else optuna_cfg.n_trials

        _log.info(
            "event='tune.start' task=%s resume=%s n_trials=%d",
            cfg.task,
            resume,
            actual_n_trials,
        )

        self._ensure_run_dir(generate_run_id())
        # Components are dropped: tune() only needs the sorted X/y/groups for
        # the objective closure; FeaturePipeline state is built per-trial via
        # `provider.build_pipeline_factory()`.
        X, y, groups, _components = self._prepare_training_data(data)
        del _components
        self._X, self._y = X, y

        provider = get_provider(cfg.model)
        self._provider = provider
        n_classes = int(y.nunique()) if cfg.task == "multiclass" else None
        splitter = build_splitter(
            cfg,
            block_values=self._block_values,
            task=cfg.task,
            seed=cfg.training.seed,
        )
        base_model_params, base_smart_params = self._merge_params(provider)

        space, used_default, fixed = self._resolve_search_space(
            resume=resume, provider=provider
        )
        space, boundary_report, expanded_names = self._maybe_expand_boundary(
            space,
            resume=resume,
            used_default=used_default,
            expand_boundary=expand_boundary,
            boundary_threshold=boundary_threshold,
        )

        # --- Metric & evaluator setup --------------------------------------------
        metric_entries = cfg.evaluation.metrics or _DEFAULT_METRICS[cfg.task]

        first_entry = metric_entries[0]
        metric_name, _ = parse_metric_entry(first_entry)

        evaluator = Evaluator(task=cfg.task)
        fingerprint = fp_compute(X, file_path=None)
        run_meta = self._build_run_meta(generate_run_id())

        objective = self._build_tune_objective(
            space=space,
            base_model_params=base_model_params,
            base_smart_params=base_smart_params,
            fixed=fixed,
            provider=provider,
            splitter=splitter,
            X=X,
            y=y,
            groups=groups,
            n_classes=n_classes,
            fingerprint=fingerprint,
            run_meta=run_meta,
            evaluator=evaluator,
            first_entry=first_entry,
            metric_name=metric_name,
        )

        round_number = self._round_number + 1
        result, study, prior_trials, best_score_before = self._run_tune_round(
            objective,
            space=space,
            actual_n_trials=actual_n_trials,
            optuna_cfg=optuna_cfg,
            metric_name=metric_name,
            progress_callback=progress_callback,
            storage=storage,
            study_name=study_name,
            resume=resume,
            round_number=round_number,
            expanded_names=expanded_names,
        )

        final_result, all_rounds = self._assemble_tuning_result(
            result,
            round_number=round_number,
            actual_n_trials=actual_n_trials,
            best_score_before=best_score_before,
            expanded_names=expanded_names,
            space=space,
            boundary_report=boundary_report,
            prior_trials=prior_trials,
        )

        # --- Update internal state -----------------------------------------------
        self._tuning_result = final_result
        self._study = study
        self._round_number = round_number
        self._rounds = list(all_rounds)
        self._space = space
        self._used_default_space = used_default

        _log.info(
            "event='tune.done' round=%d best_params=%s",
            round_number,
            final_result.best_params,
        )
        return final_result

    # --- tune() helpers (#114 — split god method) -------------------------------

    def _validate_tune_inputs(self, *, resume: bool, boundary_threshold: float) -> None:
        """Validate cfg + tune-call invariants. Raises ``LizyMLError`` on
        violation."""
        cfg = self._cfg
        if cfg.tuning is None:
            raise LizyMLError(
                code=ErrorCode.CONFIG_INVALID,
                user_message=(
                    "No tuning configuration found. "
                    "Add a 'tuning' section to the config to enable tuning."
                ),
                context={"task": cfg.task, "model": getattr(cfg.model, "name", None)},
            )

        if resume and self._study is None:
            raise LizyMLError(
                code=ErrorCode.TUNING_FAILED,
                user_message=(
                    "Cannot resume tuning: no previous tune() call. "
                    "Run tune() first, then tune(resume=True)."
                ),
                context={"resume": True, "round_number": self._round_number},
            )

        if not 0.0 < boundary_threshold < 0.5:
            raise LizyMLError(
                code=ErrorCode.CONFIG_INVALID,
                user_message=(
                    f"boundary_threshold must be in (0.0, 0.5), "
                    f"got {boundary_threshold}."
                ),
                context={"boundary_threshold": boundary_threshold},
            )

    def _resolve_search_space(
        self,
        *,
        resume: bool,
        provider: EstimatorProvider,
    ) -> tuple[list[Any], bool, dict[str, Any]]:
        """Return the search space for this tune call.

        Returns a tuple ``(space, used_default, fixed_params)`` where
        ``used_default`` signals that no user-supplied space was provided
        (drives the H-0068 expand-boundary default).

        H-0078: ``provider.parameter_bounds(task)`` is attached to each
        matching dim so that boundary expansion in subsequent rounds is
        clamped to physically-meaningful limits.
        """
        cfg = self._cfg
        assert cfg.tuning is not None  # noqa: S101 — validated upstream
        if resume and self._space is not None:
            space = list(self._space)
            used_default = self._used_default_space
        else:
            user_space = parse_space(cfg.tuning.optuna.space)
            if user_space:
                space = user_space
                used_default = False
            else:
                space = provider.default_space(cfg.task)
                used_default = True
            space = attach_bounds(space, provider.parameter_bounds(cfg.task))

        fixed: dict[str, Any] = (
            provider.default_fixed_params(cfg.task) if used_default else {}
        )
        return space, used_default, fixed

    def _maybe_expand_boundary(
        self,
        space: list[Any],
        *,
        resume: bool,
        used_default: bool,
        expand_boundary: bool | None,
        boundary_threshold: float,
    ) -> tuple[list[Any], BoundaryReport | None, tuple[str, ...]]:
        """Apply H-0068 boundary detection + expansion when applicable.

        Returns ``(new_space, boundary_report, expanded_names)``. Without
        a prior tune() or when ``expand_boundary`` is False, the input
        ``space`` is returned unchanged with ``(None, ())``.
        """
        if not (resume and self._tuning_result is not None):
            return space, None, ()

        should_expand = expand_boundary
        if should_expand is None:
            should_expand = used_default

        if not should_expand:
            _log.info("event='tune.resume' expand_boundary=False")
            return space, None, ()

        boundary_report = detect_boundary(
            space, self._tuning_result.best_params, boundary_threshold
        )
        expanded_names = boundary_report.expanded_names
        if not expanded_names:
            _log.info("event='tune.resume' no dims near boundary")
            return space, boundary_report, expanded_names

        new_space = expand_dims(space, boundary_report)
        for name in expanded_names:
            status = next(s for s in boundary_report.dims if s.name == name)
            _log.info(
                "event='tune.expand' dim=%s edge=%s old=[%s, %s] new=[%s, %s] best=%s",
                name,
                status.edge,
                status.low,
                status.high,
                status.new_low,
                status.new_high,
                status.best_value,
            )
        return new_space, boundary_report, expanded_names

    def _build_tune_objective(
        self,
        *,
        space: list[Any],
        base_model_params: dict[str, Any],
        base_smart_params: dict[str, Any],
        fixed: dict[str, Any],
        provider: EstimatorProvider,
        splitter: BaseSplitter,
        X: pd.DataFrame,
        y: pd.Series,
        groups: npt.NDArray[Any] | None,
        n_classes: int | None,
        fingerprint: DataFingerprint,
        run_meta: RunMeta,
        evaluator: Evaluator,
        first_entry: str | dict[str, Any],
        metric_name: str,
    ) -> Callable[[Any], float]:
        """Return the optuna objective closure used by ``Tuner``.

        Captures all parameters needed by a single trial: the provider,
        splitter, training data, fingerprint/run-meta and the metric to
        optimise. The closure rebuilds ``TrainComponents`` and ``CVTrainer``
        per trial because trial-level params (``smart``, ``training``)
        change per call.
        """
        cfg = self._cfg

        def objective(trial: Any) -> float:
            trial_params = suggest_params(trial, space)
            model_p, smart_p, training_p = split_by_category(trial_params, space)

            merged_model = {**base_model_params, **fixed, **model_p}
            merged_smart = {**base_smart_params, **smart_p}

            tc = self._build_train_components(
                X,
                y,
                provider=provider,
                model_params=merged_model,
                smart_params=merged_smart,
                training_overrides=training_p,
            )

            cv_trainer = CVTrainer(
                outer_splitter=splitter,
                inner_valid=tc.inner_valid,
                pipeline_factory=provider.build_pipeline_factory(),
                estimator_factory=tc.estimator_factory,
                task=cfg.task,
                n_classes=n_classes,
                ratio_param_resolver=tc.ratio_resolver,
            )
            fit_result = cv_trainer.fit(
                X,
                y,
                groups,
                data_fingerprint=fingerprint,
                run_meta=run_meta,
                sample_weight=tc.sample_weight,
            )
            metrics = evaluator.evaluate(fit_result, y, [first_entry])
            score: float = metrics["raw"]["oof"][metric_name]
            return score

        return objective

    def _run_tune_round(
        self,
        objective: Callable[[Any], float],
        *,
        space: list[Any],
        actual_n_trials: int,
        optuna_cfg: OptunaParamsConfig,
        metric_name: str,
        progress_callback: TuneProgressCallback | None,
        storage: str | BaseStorage | None,
        study_name: str | None,
        resume: bool,
        round_number: int,
        expanded_names: tuple[str, ...],
    ) -> tuple[TuningResult, Any, int, float | None]:
        """Construct the ``Tuner`` and execute one round of optuna search.

        Returns ``(raw_result, study, prior_trials, best_score_before)``.
        ``prior_trials`` and ``best_score_before`` are populated only when
        ``resume`` is True (validated upstream by ``_validate_tune_inputs``).
        """
        prior_trials = 0
        best_score_before: float | None = None
        enqueue: dict[str, Any] | None = None
        if resume:
            assert self._study is not None  # noqa: S101 — validated upstream
            assert self._tuning_result is not None  # noqa: S101
            prior_trials = len(self._study.trials)
            best_score_before = self._tuning_result.best_score
            enqueue = dict(self._tuning_result.best_params)

        tuner = Tuner(
            dims=space,
            n_trials=actual_n_trials,
            direction=optuna_cfg.direction,
            timeout=optuna_cfg.timeout,
            seed=self._cfg.training.seed,
            progress_callback=progress_callback,
            storage=storage,
            study_name=study_name,
        )
        result, study = tuner.tune(
            objective,
            metric_name=metric_name,
            study=self._study if resume else None,
            enqueue_params=enqueue,
            round_number=round_number,
            prior_trials=prior_trials,
            expanded_dims=expanded_names,
        )
        return result, study, prior_trials, best_score_before

    def _assemble_tuning_result(
        self,
        raw_result: TuningResult,
        *,
        round_number: int,
        actual_n_trials: int,
        best_score_before: float | None,
        expanded_names: tuple[str, ...],
        space: list[Any],
        boundary_report: BoundaryReport | None,
        prior_trials: int,
    ) -> tuple[TuningResult, tuple[RoundSummary, ...]]:
        """Delegate round-summary/trial-renumber assembly to ``tuning/rounds.py``.

        The domain logic (``RoundSummary`` construction, cumulative round
        boundaries, per-trial ``round`` renumbering) lives in ``tuning/`` per
        the category contract (#209); the Facade only supplies ``self._rounds``.
        """
        return assemble_round_result(
            raw_result,
            round_number=round_number,
            actual_n_trials=actual_n_trials,
            best_score_before=best_score_before,
            expanded_names=expanded_names,
            space=space,
            boundary_report=boundary_report,
            prior_trials=prior_trials,
            prior_rounds=tuple(self._rounds),
        )

    def _get_tuning_state(self) -> TuningState:
        """Return a frozen snapshot of post-tune state for tuning Mixin methods.

        Used by ``tuning_plot`` / ``tuning_table`` / ``boundary_table`` which
        operate on tuning artefacts even when ``fit()`` has not been called.
        Kept distinct from ``_get_fit_state`` to preserve the latter's
        "fit-required" invariant (H-0077).

        Raises:
            :class:`~lizyml.core.exceptions.LizyMLError` with
            ``MODEL_NOT_FIT`` when ``tune()`` has not been called.
        """
        if self._tuning_result is None:
            raise LizyMLError(
                code=ErrorCode.MODEL_NOT_FIT,
                user_message="tune() has not been called yet.",
                context={"method": "_get_tuning_state", "task": self._cfg.task},
            )
        return TuningState(cfg=self._cfg, tuning_result=self._tuning_result)
