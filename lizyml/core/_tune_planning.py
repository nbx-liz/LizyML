"""_tune_planning -- phases A to D of ``tune()``, shared with the estimate (H-0150).

``Model.tune`` and ``Model.estimate_tune_memory`` plan the next round through
this one function (policy 4), so the estimate cannot judge a round another way
than the tune that follows it (DC3). Phases A to D only read: arguments, data,
locating and judging the study, and planning the round (H-0122). Every refusal
of a tune call is raised here, before any study write (INV-R4). The caller owns
what comes after: the writes and the trials (``tune``), or the envelope (the
estimate).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from lizyml.core._feature_name_check import check_feature_names_for_training
from lizyml.core._model_factories import build_splitter, get_provider
from lizyml.core._tuning_admission import (
    check_round_space,
    check_space_names,
    parse_config_space,
    parse_space_arg,
    refuse_space_on_create,
    validate_tune_args,
)
from lizyml.core._tuning_identity import summarise_frame
from lizyml.core._tuning_plan import RoundPlan, plan_round
from lizyml.core._tuning_restore import refuse_study_smart_values
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
from lizyml.tuning import tuner as _tuner
from lizyml.tuning.lineage import Lineage, classify_study

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd
    from optuna.storages import BaseStorage

    from lizyml.config.schema import LizyMLConfig
    from lizyml.core._tuning_resume import StudyTarget
    from lizyml.data.dataframe_builder import DataFrameComponents
    from lizyml.splitters.base import BaseSplitter


@dataclass(frozen=True)
class TunePlan:
    """What phases A to D of a tune call decided."""

    direction: Literal["minimize", "maximize"]
    provider: Any
    X: pd.DataFrame
    y: pd.Series
    groups: npt.NDArray[Any] | None
    n_classes: int | None
    splitter: BaseSplitter
    target: StudyTarget
    lineage: Lineage | None
    plan: RoundPlan
    base_model_params: dict[str, Any]
    base_smart_params: dict[str, Any]
    study_payload: Any
    round_payload: Any
    held_storage: Any
    held_name: str | None


class TunePlanningMixin:
    """Phases A to D of ``tune()``, as one function."""

    if TYPE_CHECKING:
        _cfg: LizyMLConfig
        _study: Any
        _round_number: int
        _block_values: npt.NDArray[Any] | None

        def _load_data(self, data: pd.DataFrame | None) -> pd.DataFrame: ...

        def _tune_splitter(
            self, build: Callable[..., BaseSplitter]
        ) -> BaseSplitter: ...

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
    ) -> TunePlan:
        """Run phases A to D of a tune call; write nothing to the study."""
        cfg = self._cfg
        # --- Phase A: arguments (no I/O) --------------------------------------
        validate_tune_args(
            cfg,
            boundary_threshold=boundary_threshold,
            storage=storage,
            study_name=study_name,
        )
        # The config space's format before anything reads a raw spec (H-0123).
        parse_config_space(cfg)
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
        # H-0148: before the study is read or written and before any trial.
        check_feature_names_for_training(provider, X)
        n_classes = int(y.nunique()) if cfg.task == "multiclass" else None
        # Built through a hook in `_model_tuning`, so the H-0013 kfold
        # warning (stacklevel=2) keeps naming that module (H-0150).
        splitter = self._tune_splitter(build_splitter)

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
                # This frame sits between `tune` and the check, so the
                # version warning reaches one frame further (finding 6).
                stacklevel=5,
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
        refuse_study_smart_values(lineage, trials, provider, cfg.task, shown_name)
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
        return TunePlan(
            direction=direction,
            provider=provider,
            X=X,
            y=y,
            groups=groups,
            n_classes=n_classes,
            splitter=splitter,
            target=target,
            lineage=lineage,
            plan=plan,
            base_model_params=base_model_params,
            base_smart_params=base_smart_params,
            study_payload=study_payload,
            round_payload=round_payload,
            held_storage=held_storage,
            held_name=held_name,
        )
