"""ModelTuningMixin — tune() orchestration extracted from the Model facade (H-0091).

Writer-exempt *orchestrator* mixin. Unlike the read-only *diagnostic* mixins
(``_model_plots`` / ``_model_tables`` / ``_model_persistence``, whose read-only
contract is enforced by ``tests/test_core/test_mixin_state_isolation.py``), this
mixin runs during the mutating ``tune()`` lifecycle and deliberately reads and
writes Model body state (``self._tuning_result`` / ``_study`` / ``_rounds`` /
``_round_number`` / ``_space`` / ``_used_default_space`` ...). It is
intentionally **not** listed in that guard's ``_MIXIN_FILES``; the read-only
H-0077 invariant applies only to the diagnostic mixins. See HISTORY H-0091 (#237).

``tune()`` runs in phases (H-0122): A arguments, B data, C locate and judge the
study, D plan the round -- these only read, and every refusal is raised in
them -- then E the study writes and the trials, and F the Model commit. So a
refused call changes neither the study nor the Model (INV-R4).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from lizyml.core._model_factories import (
    get_outer_n_splits,
    get_provider,
)
from lizyml.core._model_state import TuningState
from lizyml.core._resource_warning import WarningLatch, check_resources
from lizyml.core._tune_budget import (
    TuneBudget,
    admit_budget,
    build_trial,
    validate_budget_value,
)
from lizyml.core._tune_round_close import close_round
from lizyml.core._tune_support import (
    _first_metric,
    _restored_on_failure,
    _trial_errors,
    components_builder,
)
from lizyml.core._tuning_admission import (
    validate_tune_args,
)
from lizyml.core._tuning_identity import summarise_frame
from lizyml.core._tuning_plan import RoundPlan
from lizyml.core._tuning_restore import (
    read_restorable,
)
from lizyml.core._tuning_space import space_summary
from lizyml.core._tuning_validation import resolve_tuning_direction
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.logging import generate_run_id, get_logger
from lizyml.core.train_components import TrainComponents
from lizyml.core.types.artifacts import RunMeta
from lizyml.core.types.tuning_result import (
    RoundSummary,
    TuneProgressCallback,
    TuningResult,
)
from lizyml.data.fingerprint import compute as fp_compute
from lizyml.evaluation.evaluator import Evaluator
from lizyml.training.cv_trainer import CVTrainer
from lizyml.tuning import tuner as _tuner
from lizyml.tuning.lineage import (
    STUDY_KEY,
    StoredRoundEnd,
    classify_study,
    encode_round_end,
    end_key,
    round_key,
)
from lizyml.tuning.rounds import assemble_tuning_result, recorded_trial_params
from lizyml.tuning.search_space import suggest_params
from lizyml.tuning.study_lineage import require_stored_study_args
from lizyml.tuning.tuner import Tuner

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt
    import pandas as pd
    from optuna.storages import BaseStorage

    from lizyml.config.schema import LizyMLConfig
    from lizyml.core._tune_planning import TunePlan
    from lizyml.core.types.artifacts import DataFingerprint
    from lizyml.core.types.search_dim import SearchDim
    from lizyml.data.dataframe_builder import DataFrameComponents
    from lizyml.estimators.provider import EstimatorProvider
    from lizyml.splitters.base import BaseSplitter

_log = get_logger("model")


def _interrupted(plan: RoundPlan) -> int:
    """Trials of *plan*'s own round this call moved from RUNNING to FAIL: those
    of an open round being finished; a replacement's belong to the old round."""
    return len(plan.running_to_fail) if plan.kind == "open" else 0


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
        _tuning_history_available: bool
        _y: pd.Series | None
        _X: pd.DataFrame | None
        _provider: EstimatorProvider | None
        _block_values: npt.NDArray[Any] | None
        _study: Any
        _round_number: int
        _rounds: list[RoundSummary]
        _space: list[Any] | None
        _used_default_space: bool
        _tuning_fixed_params: dict[str, Any] | None
        _tuning_storage: str | BaseStorage | None
        _tuning_study_name: str | None

        # --- Facade methods this mixin delegates to (defined on Model) ---
        def _load_data(self, data: pd.DataFrame | None) -> pd.DataFrame: ...

        def _prepare_training_data(
            self, data: pd.DataFrame | None
        ) -> tuple[
            pd.DataFrame, pd.Series, npt.NDArray[Any] | None, DataFrameComponents
        ]: ...

        def _merge_params(
            self,
            provider: Any,
            override: dict[str, Any] | None = None,
            *,
            tuning_fixed_params: dict[str, Any] | None = None,
            include_tuning_result: bool = True,
        ) -> tuple[dict[str, Any], dict[str, Any]]: ...

        def _ensure_run_dir(self, run_id: str) -> None: ...

        def _plan_tune(
            self,
            data: pd.DataFrame | None,
            *,
            resume: bool,
            n_trials: int | None,
            expand_boundary: bool | None,
            boundary_threshold: float,
            storage: str | BaseStorage | None,
            study_name: str | None,
            space: Mapping[str, Any] | None,
        ) -> TunePlan: ...

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
        space: Mapping[str, Any] | None = None,
        memory_budget_bytes: int | None = None,
    ) -> TuningResult:
        """Run one round of hyperparameter search with optuna (H-0068, H-0072,
        H-0122).

        Requires ``tuning`` section in the config.  Best params are stored
        internally and used automatically in the next ``fit()`` call.

        A study records its objective identity and, per round, its space and
        budget (``lizyml:`` study attributes). Every trial of a study observes
        the same objective: a resume whose data, config outside the search
        space, split or first metric differs is refused. Each round's trials
        train on ``config ⊕ fixed ⊕ the sampled values``; the previous best is
        not re-evaluated.

        Args:
            data: Training DataFrame.  Overrides any data from construction
                or ``data.path`` in config.
            resume: If True, add a round to a study: the one this Model holds,
                or, when it holds none, the stored ``(storage, study_name)``
                study (created when absent). A round left open by a crash is
                finished with its recorded budget instead.
            n_trials: Number of trials to run.  Defaults to
                ``config.tuning.optuna.params.n_trials`` if None. An open
                round only accepts its recorded value, unless a different
                ``space=`` replaces it (below).
            expand_boundary: Whether to auto-expand search space dimensions
                whose best params are near the boundary.  None means True
                for default space, False for user-specified space. An open
                round only accepts its recorded value, unless a different
                ``space=`` replaces it (below).
            boundary_threshold: Edge detection threshold; must be in the open
                interval ``(0.0, 0.5)``.
            progress_callback: Optional callback invoked after each trial.
            storage: Optional Optuna storage URL or ``BaseStorage`` instance
                for resumable tuning (H-0072). ``None`` keeps the in-memory
                behavior. When set, ``study_name`` is required.
            study_name: Study identifier used together with ``storage``, and
                only with it. With ``resume=False`` the name must be new in
                the storage; continuing a stored study takes ``resume=True``.
            space: This round's search space, in the form of
                ``tuning.optuna.space``, resolved by the study's
                ``space_mode`` (``merge`` lays it over the provider's
                defaults; ``replace`` declares exactly it). Only for a round
                after the first (``resume=True`` on a study with a round);
                ``None`` keeps the previous round's space. A dimension whose
                declaration changes is not expanded this round; its
                distribution kind, ``log`` and choices cannot change within a
                study. An open round only accepts its recorded space.
            memory_budget_bytes: Opt-in cap on the extra CPU memory of one
                trial (H-0150). ``None`` (default) changes nothing. A
                positive ``int`` refuses, before it trains, each trial whose
                estimate is over it (equal passes), whose assessment raises,
                or whose finalized device is not the CPU; the trial is
                recorded FAIL (``event='tune.trial.memory_refused'`` is
                logged, and the trial error carries, its ``reason`` and the
                budget). It is compared with the estimate of extra CPU
                memory, not the process's resident set. It is per call:
                never stored, and it may change on resume. A round that can
                reach a non-CPU device, or a provider that cannot estimate
                memory or the adapter it builds, is refused before any
                write; a round that runs no trial closes as without a budget.

        Returns:
            :class:`~lizyml.core.types.tuning_result.TuningResult` with
            best params, best score, and full trial history.

        Raises:
            LizyMLError with CONFIG_INVALID when no ``tuning`` config is set,
                for a ``tuning.optuna.space`` dimension outside the documented
                format (H-0123),
                for ``storage`` / ``study_name`` given alone, for
                ``resume=False`` onto a stored study, for a target that is not
                the held study, for a study whose record is missing,
                unreadable or inconsistent, or whose objective or config space
                differs from this call's, and for a ``space=`` that is
                outside the same format, given to a call that creates a study, changes a
                dimension's category, distribution kind, ``log`` or choices,
                or differs from the recorded space of an open round that
                holds a COMPLETE trial. An open round with no COMPLETE
                trial is instead closed ``failed`` (``UserWarning``) and a
                new round starts with the given space, ``n_trials`` and
                ``expand_boundary`` (H-0128).
            LizyMLError with OPTIONAL_DEP_MISSING when optuna is not installed.
            LizyMLError with TUNING_FAILED on study failure, ``resume=True``
                with neither a held study nor a storage, or a round none of
                whose own trials completed (H-0128; context ``trial_states``
                and, when known, ``first_error``; ``memory_refused_this_call``
                and ``memory_budget_bytes`` describe this call only, so a
                resumed round whose earlier trials were all refused reports 0).
            LizyMLError with CONFIG_INVALID for a ``memory_budget_bytes``
                that is not ``None`` or a positive ``int``, a budgeted round
                that runs trials but whose provider cannot estimate memory
                or the adapter it builds, or whose device may not be the CPU.
        """
        cfg = self._cfg
        budget = TuneBudget(validate_budget_value(memory_budget_bytes))
        # --- Phases A-D: arguments, data, locate and judge the study, plan -----
        # Shared with `Model.estimate_tune_memory` (H-0150 policy 4); they only
        # read, and every refusal of the call is raised in them (INV-R4).
        ctx = self._plan_tune(
            data,
            resume=resume,
            n_trials=n_trials,
            expand_boundary=expand_boundary,
            boundary_threshold=boundary_threshold,
            storage=storage,
            study_name=study_name,
            space=space,
        )
        direction, provider, splitter = ctx.direction, ctx.provider, ctx.splitter
        X, y, groups, n_classes = ctx.X, ctx.y, ctx.groups, ctx.n_classes
        target, lineage, plan = ctx.target, ctx.lineage, ctx.plan
        base_model_params = ctx.base_model_params
        base_smart_params = ctx.base_smart_params
        study_payload, round_payload = ctx.study_payload, ctx.round_payload
        held_storage, held_name = ctx.held_storage, ctx.held_name
        if budget.limit is not None:  # before any write (policy 5, 6a)
            admit_budget(
                provider,
                runs_optimize=plan.runs_optimize,
                dims=plan.effective,
                fixed=dict(plan.fixed),
                base_model_params=base_model_params,
                base_smart_params=base_smart_params,
                n_boosters=get_outer_n_splits(cfg),
                build_components=components_builder(self, X, y, provider),
                X=X,
            )

        # --- Phase E: writes, in the §7.1 order --------------------------------
        # A tune that fails -- an exception, a dying process, or a round that
        # closes `failed` -- leaves these two as they were too (§7.1, H-0094
        # decision 14); the run directory it created on disk may remain.
        with _restored_on_failure(self, ("_run_dir", "_provider")):
            self._ensure_run_dir(generate_run_id())
            self._provider = provider
            first_entry, metric_name = _first_metric(cfg)
            objective = self._build_tune_objective(
                space=list(plan.effective),
                base_model_params=base_model_params,
                base_smart_params=base_smart_params,
                fixed=dict(plan.fixed),
                provider=provider,
                splitter=splitter,
                X=X,
                y=y,
                groups=groups,
                n_classes=n_classes,
                fingerprint=fp_compute(X, file_path=None),
                run_meta=self._build_run_meta(generate_run_id()),
                evaluator=Evaluator(task=cfg.task),
                first_entry=first_entry,
                metric_name=metric_name,
                budget=budget,
            )
            _log.info(
                "event='tune.start' task=%s resume=%s round=%d plan=%s n_trials=%d",
                cfg.task,
                resume,
                plan.round_number,
                plan.kind,
                plan.run_n_trials,
            )
            tuner = Tuner(
                dims=list(plan.effective),
                n_trials=plan.run_n_trials,
                direction=direction,
                timeout=plan.run_timeout,
                seed=cfg.training.seed,
                progress_callback=progress_callback,
                storage=storage,
                study_name=study_name,
            )
            if target.announce_create:
                _log.info(
                    "event='tune.study_created' reason=resume_without_stored_study "
                    "study_name=%s",
                    study_name,
                )
            study = target.study
            if study is None:  # (a) create
                study = _tuner.create_study(
                    direction, cfg.training.seed, storage=storage, study_name=study_name
                )
            if study_payload is not None:  # (b) the study record
                _tuner.write_study_attr(study, STUDY_KEY, study_payload)
            if plan.running_to_fail:  # left RUNNING by a dead process
                _tuner.fail_running_trials(study, plan.running_to_fail)
            if plan.replaces_round is not None:  # H-0128 policy 7 (b)
                assert lineage is not None and lineage.last_round is not None  # noqa: S101
                recorded_space = space_summary(
                    lineage.last_round.declared, lineage.routes
                )
                given_space = space_summary(plan.declared, lineage.routes)
                warnings.warn(
                    f"Round {plan.replaces_round} had no completed trial and was "
                    "started with another space=; it is closed failed and round "
                    f"{plan.round_number} starts with the given space= "
                    f"(recorded: {recorded_space}; given: {given_space}).",
                    UserWarning,
                    stacklevel=3,
                )
                _tuner.write_study_attr(
                    study,
                    end_key(plan.replaces_round),
                    encode_round_end(
                        StoredRoundEnd(
                            "failed", None, datetime.now(tz=timezone.utc).isoformat()
                        )
                    ),
                )
            if round_payload is not None:  # (2) the round record
                _tuner.write_study_attr(
                    study, round_key(plan.round_number), round_payload
                )
            if plan.runs_optimize:  # (3) the trials
                self._run_tune_round(tuner, study, objective, plan)
            else:
                _log.info(
                    "event='tune.resume_open' round=%d budget_spent=true "
                    "remaining_trials=%d remaining_timeout=%s",
                    plan.round_number,
                    plan.run_n_trials,
                    plan.run_timeout,
                )
            self._close_round(
                study, plan, first_error=tuner.first_error, budget=budget
            )  # (4)

            # --- Phase F: assemble from the lineage (the commit follows) ---------
            final_trials = study.trials
            final = classify_study(study.user_attrs, final_trials)
            if final.state != "completed":
                raise LizyMLError(
                    code=ErrorCode.TUNING_FAILED,
                    user_message=(
                        f"Internal error: round {plan.round_number} closed but the "
                        f"study's lineage reads '{final.state}' ({final.reason})."
                    ),
                    context={"round": plan.round_number, "state": final.state},
                )
            best = study.best_trial
            result = assemble_tuning_result(
                final,
                final_trials,
                recorded_trial_params(final, best),
                best.value,
                metric_name=metric_name,
                direction=direction,
                current_round=plan.round_number,
            )
        # Published together, and only once the round has closed. A tune that
        # fails leaves every field as it was, including `_X` / `_y`, which
        # describe the retained fit (H-0094 decision 14).
        if target.kind != "continue":
            held_storage, held_name = storage, study_name
        self._commit_tuning(
            X=X,
            y=y,
            result=result,
            study=study,
            round_number=plan.round_number,
            effective=plan.effective,
            used_default=plan.used_default,
            fixed=plan.fixed,
            storage=held_storage,
            study_name=held_name,
        )

        _log.info(
            "event='tune.done' round=%d best_params=%s",
            plan.round_number,
            result.best_params,
        )
        return result

    def restore_tuning(
        self,
        storage: str | BaseStorage,
        study_name: str,
        data: pd.DataFrame | None = None,
    ) -> TuningResult:
        """Return the result of a stored study's last round without adding to
        it (H-0122 §7.7).

        For a tune whose round closed but whose caller never received the
        result (the process died in between): the study is in state
        ``completed``, and ``tune(resume=True)`` would start a new round. The
        result is the one that tune returned, and the Model is left as that
        tune left it, so ``fit()`` trains with the best trial's parameters
        (and the study's ``fixed``) and a later ``tune(resume=True)``
        continues this study. A fitted Model keeps its fit; only the tuning
        result is replaced. Nothing is written to the study.

        Args:
            storage: Optuna storage URL or ``BaseStorage`` holding the study.
            study_name: The study's name in *storage*.
            data: Training DataFrame, with the same priority as in
                :meth:`tune`. It must be the data the study observes.

        Returns:
            The :class:`~lizyml.core.types.tuning_result.TuningResult` of the
            study's last round.

        Raises:
            LizyMLError with CONFIG_INVALID when no ``tuning`` config is set,
                when ``storage`` or ``study_name`` is missing, when the study
                has an open (interrupted) round, when its record is missing,
                unreadable or inconsistent, or when its direction, objective
                or config space differs from this Model's.
            LizyMLError with TUNING_FAILED when the storage has no such study,
                or the study has no completed round to restore.
            LizyMLError with OPTIONAL_DEP_MISSING when optuna is not installed.
        """
        cfg = self._cfg
        # The tuning-config and argument checks of `tune`; the threshold is
        # its default, since a restore reports the recorded one.
        validate_tune_args(
            cfg,
            boundary_threshold=0.05,
            storage=storage,
            study_name=study_name,
            caller="restore_tuning",
        )
        require_stored_study_args(storage, study_name, caller="restore_tuning")
        _tuner.require_optuna()  # before data or storage (extras registry)
        direction = resolve_tuning_direction(cfg)
        raw = self._load_data(data)
        current_data = summarise_frame(raw)
        X, y, _groups, _components = self._prepare_training_data(raw)
        restored = read_restorable(
            cfg,
            storage,
            study_name,
            data_summary=current_data,
            provider=get_provider(cfg.model),
            direction=direction,
            metric_name=_first_metric(cfg)[1],
        )
        self._commit_tuning(
            X=X,
            y=y,
            result=restored.result,
            study=restored.study,
            round_number=restored.round_number,
            effective=restored.effective,
            used_default=restored.used_default,
            fixed=restored.fixed,
            storage=storage,
            study_name=study_name,
        )
        _log.info(
            "event='tune.restored' round=%d study_name=%s",
            restored.round_number,
            study_name,
        )
        return restored.result

    def _commit_tuning(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        result: TuningResult,
        study: Any,
        round_number: int,
        effective: Sequence[SearchDim],
        used_default: bool,
        fixed: Mapping[str, Any],
        storage: str | BaseStorage | None,
        study_name: str | None,
    ) -> None:
        """Publish a closed round's committed tune state, all fields together.

        Shared by ``tune`` and ``restore_tuning`` (H-0094 decisions 13 and
        14): the fit's fields are not touched, and a call that fails before
        this leaves these fields as they were. (Per-call transients such as
        ``_provider`` / ``_run_dir`` in ``tune`` are handled at the call.)
        """
        self._X, self._y = X, y
        self._tuning_result = result
        # A result tune / restore_tuning built holds its whole history (H-0134 P5).
        self._tuning_history_available = True
        self._study = study
        self._round_number = round_number
        self._rounds = list(result.rounds)
        self._space = list(effective)
        self._used_default_space = used_default
        self._tuning_fixed_params = dict(fixed)
        self._tuning_storage = storage
        self._tuning_study_name = study_name

    # --- tune() helpers (#114 — split god method) -------------------------------

    def _tune_splitter(self, build: Callable[..., BaseSplitter]) -> BaseSplitter:
        """The outer splitter of a tune call; *build* is ``build_splitter``.

        Called from here, so the H-0013 kfold warning (``stacklevel=2``) names
        this module as it did before the planning moved out (H-0150).
        """
        cfg = self._cfg
        return build(
            cfg, block_values=self._block_values, task=cfg.task, seed=cfg.training.seed
        )

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
        budget: TuneBudget | None = None,
    ) -> Callable[[Any], float]:
        """Return the optuna objective closure used by ``Tuner``.

        Captures all parameters needed by a single trial: the provider,
        splitter, training data, fingerprint/run-meta and the metric to
        optimise. The closure rebuilds ``TrainComponents`` and ``CVTrainer``
        per trial because trial-level params (``smart``, ``training``)
        change per call.
        """
        cfg = self._cfg
        # H-0140: one warning per tune() call (this closure is built once per
        # call); the later trials that fire are logged.
        latch = WarningLatch()
        limits = budget if budget is not None else TuneBudget()
        n_folds = get_outer_n_splits(cfg)

        def objective(trial: Any) -> float:
            trial_params = suggest_params(trial, space)
            # The trial's parameters (`trial_inputs`, shared with the memory
            # estimates) and components, built once. With a budget the device
            # and the estimate are read from these components, and the trial
            # is refused before it trains (H-0150 policy 5, 6 (b)).
            built = build_trial(
                limits,
                provider,
                space,
                trial_params,
                base_model_params=base_model_params,
                base_smart_params=base_smart_params,
                fixed=fixed,
                n_boosters=n_folds,
                build_components=components_builder(self, X, y, provider),
                X=X,
            )
            tc = built.components
            # Before the trial trains; its outer folds are kept alive together.
            check_resources(
                provider,
                tc,
                X,
                built.smart,
                n_boosters=n_folds,
                stacklevel=2,
                latch=latch,
            )

            cv_trainer = CVTrainer(
                outer_splitter=splitter,
                inner_valid=tc.inner_valid,
                pipeline_factory=provider.build_pipeline_factory(
                    unseen_policy=cfg.features.unseen_policy
                ),
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
        tuner: Tuner,
        study: Any,
        objective: Callable[[Any], float],
        plan: RoundPlan,
    ) -> None:
        """Add the round's trials to *study* (write position 3).

        An exception -- or a ``BaseException`` such as a dying process --
        propagates without the round's end record: the round stays open and
        the next ``resume=True`` finishes it (§7.2).
        """
        tuner.run(
            study,
            objective,
            round_number=plan.round_number,
            round_start=plan.start_trial,
            progress_total=plan.budget_n_trials,
            expanded_dims=plan.expanded_names,
            # H-0128 policy 5: an optional provider capability; tuning/
            # itself knows no estimator library.
            extra_catch=_trial_errors(self._provider),
            interrupted=_interrupted(plan),
        )

    def _close_round(
        self,
        study: Any,
        plan: RoundPlan,
        *,
        first_error: str | None = None,
        budget: TuneBudget | None = None,
    ) -> None:
        """Write the round end record (write position 4); see ``close_round``."""
        close_round(
            study,
            plan,
            first_error=first_error,
            budget=budget,
            interrupted=_interrupted(plan),
        )

    def _get_tuning_state(self, method: str) -> TuningState:
        """Return a frozen snapshot of post-tune state for tuning Mixin methods.

        Used by ``tuning_plot`` / ``tuning_table`` / ``boundary_table`` which
        operate on tuning artefacts even when ``fit()`` has not been called.
        Kept distinct from ``_get_fit_state`` to preserve the latter's
        "fit-required" invariant (H-0077).

        Args:
            method: The reporting surface asking, named in the error context.

        Raises:
            :class:`~lizyml.core.exceptions.LizyMLError` with
            ``MODEL_NOT_FIT`` when ``tune()`` has not been called, or when the
            model was loaded from an artifact that does not carry the trial
            history (H-0134 P6): the overlay is there, the history is not, and
            an empty table would read as a study that ran no trials.
        """
        if self._tuning_result is None:
            raise LizyMLError(
                code=ErrorCode.MODEL_NOT_FIT,
                user_message="tune() has not been called yet.",
                context={"method": method, "task": self._cfg.task},
            )
        if not self._tuning_history_available:
            raise LizyMLError(
                code=ErrorCode.MODEL_NOT_FIT,
                user_message=(
                    f"{method}() cannot answer: this model was loaded from an "
                    "artifact that does not carry the tuning trial history (it was "
                    "saved before format_version 4, or exported from a model loaded "
                    "from such an artifact). The tuned parameters are restored and "
                    "fit() uses them. If the study's storage still exists, "
                    "restore_tuning(storage=..., study_name=...) rebuilds the history."
                ),
                context={
                    "method": method,
                    "missing": "tuning_history",
                    "loaded_from_artifact": True,
                },
            )
        return TuningState(cfg=self._cfg, tuning_result=self._tuning_result)
