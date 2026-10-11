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

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from lizyml.core._model_factories import (
    build_splitter,
    get_provider,
    overlay_params,
)
from lizyml.core._model_metrics import _DEFAULT_METRICS
from lizyml.core._model_state import TuningState
from lizyml.core._tuning_admission import (
    check_round_space,
    check_space_names,
    parse_space_arg,
    refuse_space_on_create,
    validate_tune_args,
)
from lizyml.core._tuning_identity import summarise_frame
from lizyml.core._tuning_plan import RoundPlan, plan_round
from lizyml.core._tuning_restore import read_restorable
from lizyml.core._tuning_resume import (
    judge_existing,
    locate_study,
    objective_identity,
    refuse_state,
    resolve_call_row,
    round_record_payload,
    study_record_payload,
)
from lizyml.core._tuning_space import SPACE_ARG, admit_space_routes
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
from lizyml.metrics.registry import parse_metric_entry
from lizyml.training.cv_trainer import CVTrainer
from lizyml.tuning import tuner as _tuner
from lizyml.tuning.lineage import (
    STUDY_KEY,
    Lineage,
    StoredRoundEnd,
    classify_study,
    encode_round_end,
    end_key,
    round_key,
)
from lizyml.tuning.rounds import assemble_tuning_result
from lizyml.tuning.search_space import split_by_category, suggest_params
from lizyml.tuning.study_lineage import require_stored_study_args
from lizyml.tuning.tuner import Tuner

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt
    import pandas as pd
    from optuna.storages import BaseStorage

    from lizyml.config.schema import LizyMLConfig
    from lizyml.core.types.artifacts import DataFingerprint
    from lizyml.core.types.search_dim import SearchDim
    from lizyml.data.dataframe_builder import DataFrameComponents
    from lizyml.estimators.provider import EstimatorProvider
    from lizyml.splitters.base import BaseSplitter

_log = get_logger("model")


@contextmanager
def _restored_on_failure(obj: object, fields: tuple[str, ...]) -> Iterator[None]:
    """Put *fields* of *obj* back if the block leaves by any exception.

    ``BaseException`` included: a process dying inside the trials must not
    leave the Model pointing at the failed call's run directory or provider.
    """
    saved = [(name, getattr(obj, name)) for name in fields]
    try:
        yield
    except BaseException:
        for name, value in saved:
            setattr(obj, name, value)
        raise


def _first_metric(cfg: LizyMLConfig) -> tuple[str | dict[str, Any], str]:
    """The first metric's config entry and name: what tuning optimises."""
    first_entry = (cfg.evaluation.metrics or _DEFAULT_METRICS[cfg.task])[0]
    metric_name, _ = parse_metric_entry(first_entry)
    return first_entry, metric_name


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
                round only accepts its recorded value.
            expand_boundary: Whether to auto-expand search space dimensions
                whose best params are near the boundary.  None means True
                for default space, False for user-specified space. An open
                round only accepts its recorded value.
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

        Returns:
            :class:`~lizyml.core.types.tuning_result.TuningResult` with
            best params, best score, and full trial history.

        Raises:
            LizyMLError with CONFIG_INVALID when no ``tuning`` config is set,
                for ``storage`` / ``study_name`` given alone, for
                ``resume=False`` onto a stored study, for a target that is not
                the held study, for a study whose record is missing,
                unreadable or inconsistent, or whose objective or config space
                differs from this call's, and for a ``space=`` that is
                malformed, given to a call that creates a study, changes a
                dimension's category, distribution kind, ``log`` or choices,
                or differs from an open round's recorded space.
            LizyMLError with OPTIONAL_DEP_MISSING when optuna is not installed.
            LizyMLError with TUNING_FAILED on study failure, or ``resume=True``
                with neither a held study nor a storage.
        """
        cfg = self._cfg
        # --- Phase A: arguments (no I/O) --------------------------------------
        validate_tune_args(
            cfg,
            boundary_threshold=boundary_threshold,
            storage=storage,
            study_name=study_name,
        )
        # `space=` before the call matrix: an argument error wins over the
        # row-8 TUNING_FAILED, and a `resume=False` call is refused before any
        # storage is contacted (H-0122 design D-c).
        space_plain, space_dims = (
            parse_space_arg(space) if space is not None else (None, None)
        )
        refuse_space_on_create(space is not None, resume=resume)
        direction = resolve_tuning_direction(cfg)
        # Read with getattr: a Model pickled before H-0122 lacks the fields;
        # its study has no record either, so it reaches state L and is refused.
        held_storage = getattr(self, "_tuning_storage", None)
        held_name = getattr(self, "_tuning_study_name", None)
        resolve_call_row(
            resume=resume,
            has_study=self._study is not None,
            storage=storage,
            study_name=study_name,
            remembered_storage=held_storage,
            remembered_name=held_name,
            round_number=self._round_number,
            remembered_known=hasattr(self, "_tuning_storage"),
        )
        _tuner.require_optuna()

        # --- Phase B: data (per-call transients only) --------------------------
        # Summarised as loaded, before the builder casts or sorts anything, so
        # every route (argument, constructor, data.path) is judged by content.
        raw = self._load_data(data)
        current_data = summarise_frame(raw)
        X, y, groups, _components = self._prepare_training_data(raw)
        del _components
        provider = get_provider(cfg.model)
        n_classes = int(y.nunique()) if cfg.task == "multiclass" else None
        splitter = build_splitter(
            cfg,
            block_values=self._block_values,
            task=cfg.task,
            seed=cfg.training.seed,
        )

        # --- Phase C: locate and judge the study (reads only) -----------------
        target = locate_study(
            resume=resume,
            current_study=self._study,
            storage=storage,
            study_name=study_name,
            seed=cfg.training.seed,
        )
        shown_name = held_name if target.kind == "continue" else study_name
        lineage: Lineage | None = None
        trials: list[Any] = []
        if target.study is not None:
            trials = list(target.study.trials)
            lineage = classify_study(target.study.user_attrs, trials)
            refuse_state(lineage, shown_name, len(trials))
        refuse_space_on_create(
            space is not None,
            resume=resume,
            target_kind=target.kind,
            state=lineage.state if lineage is not None else None,
        )
        identity, versions, current_fixed = objective_identity(
            cfg, current_data, provider, direction
        )
        if lineage is not None:
            judge_existing(
                target.study,
                lineage,
                cfg,
                identity=identity,
                versions=versions,
                direction=direction,
                study_name=shown_name,
            )
        recorded = lineage.study if lineage is not None else None

        # --- Phase D: plan the round (pure; the remaining refusals) ------------
        # The study's `fixed` wins over the Model field (§5.3). Its value is the
        # one the record's `config.fixed` element digests (`decode_study`
        # refuses it otherwise), and identity proved that element equal to
        # this call's -- so it equals the recomputed value up to the
        # identity's canonical form (a list and a tuple are the same).
        fixed = dict(recorded.fixed) if recorded is not None else current_fixed
        check_space_names(provider, cfg)
        if space_dims is not None:
            assert lineage is not None  # noqa: S101 -- C-s1 refused E / N / U3
            # The category rule first (4.1b), so a smart or training dimension
            # re-declared as `model` is told so, not "unknown parameter".
            admit_space_routes(space_dims, lineage.routes, provider)
            check_space_names(provider, cfg, space=space_plain, surface=SPACE_ARG)
        base_model_params, base_smart_params = self._merge_params(
            provider, tuning_fixed_params=fixed, include_tuning_result=False
        )
        plan = plan_round(
            cfg,
            provider,
            lineage=lineage,
            trials=trials,
            study=target.study,
            fixed=fixed,
            base_smart=base_smart_params,
            n_trials=n_trials,
            expand_boundary=expand_boundary,
            boundary_threshold=boundary_threshold,
            space_dims=space_dims,
        )
        check_round_space(cfg, plan, provider, base_model_params, base_smart_params)
        study_payload = (
            study_record_payload(
                cfg,
                identity=identity,
                direction=direction,
                versions=versions,
                fixed=fixed,
            )
            if lineage is None or lineage.state == "uninitialized"
            else None
        )
        round_payload = round_record_payload(plan) if plan.kind != "open" else None
        # INV-R4: nothing above writes, and nothing below refuses.

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
            self._close_round(study, plan)  # (4) the end record

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
                best.params,
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
            cfg, boundary_threshold=0.05, storage=storage, study_name=study_name
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
        self._study = study
        self._round_number = round_number
        self._rounds = list(result.rounds)
        self._space = list(effective)
        self._used_default_space = used_default
        self._tuning_fixed_params = dict(fixed)
        self._tuning_storage = storage
        self._tuning_study_name = study_name

    # --- tune() helpers (#114 — split god method) -------------------------------

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

            # Overlaid by identity, the same way `_merge_params` overlays its
            # three layers. A plain dict merge keeps both spellings: a config
            # `learning_rate` and a search dimension named `eta` are one
            # parameter to LightGBM, which then prefers the canonical name --
            # so the trials trained at the config's value while the study
            # recorded the trial's, and the fit afterwards used the recorded
            # one. Tuning selected a model it had never evaluated (H-0094,
            # review round 11). This is the fourth seam; the other three were
            # made identity-aware in round 3.
            merged_model = overlay_params(provider, base_model_params, fixed)
            merged_model = overlay_params(provider, merged_model, model_p)
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
        )

    def _close_round(self, study: Any, plan: RoundPlan) -> None:
        """Write ``lizyml:round:<k>:end`` (write position 4).

        ``completed`` when the study holds a COMPLETE trial, else ``failed``
        followed by ``TUNING_FAILED``: the Model is left unchanged, and the
        next ``resume=True`` starts a new round with nothing to expand (§7.1).
        """
        complete = any(t.state.name == "COMPLETE" for t in study.trials)
        ended_at = datetime.now(tz=timezone.utc).isoformat()
        end = (
            StoredRoundEnd("completed", float(study.best_value), ended_at)
            if complete
            else StoredRoundEnd("failed", None, ended_at)
        )
        _tuner.write_study_attr(
            study, end_key(plan.round_number), encode_round_end(end)
        )
        if not complete:
            raise LizyMLError(
                code=ErrorCode.TUNING_FAILED,
                user_message="All tuning trials failed. Check parameter ranges.",
                context={
                    "round": plan.round_number,
                    "n_trials": plan.budget_n_trials,
                    "status": "failed",
                },
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
