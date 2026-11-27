"""_model_tune_memory -- ``Model.estimate_tune_memory`` (H-0150).

The estimate plans the next round through the same function as ``tune()``
(``_plan_tune``, phases A to D), so a call ``tune`` would refuse, or plan
differently, is refused or planned the same way here. It writes nothing to the
study, the trials or their attributes, and leaves the Model unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from lizyml.core._model_factories import get_outer_n_splits, get_provider
from lizyml.core._model_memory import require_memory_provider
from lizyml.core._tune_memory_envelope import (
    device_may_leave_cpu,
    evaluate_envelope,
)
from lizyml.core.train_components import TrainComponents
from lizyml.core.types.memory_estimate import TuneMemoryEstimate

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd
    from optuna.storages import BaseStorage

    from lizyml.config.schema import LizyMLConfig
    from lizyml.core._tune_planning import TunePlan


class ModelTuneMemoryMixin:
    """``Model.estimate_tune_memory``."""

    if TYPE_CHECKING:
        _cfg: LizyMLConfig
        _block_values: npt.NDArray[Any] | None
        _input_categories: dict[str, list[Any]]

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

    def estimate_tune_memory(
        self,
        data: pd.DataFrame | None = None,
        *,
        resume: bool = False,
        n_trials: int | None = None,
        expand_boundary: bool | None = None,
        boundary_threshold: float = 0.05,
        storage: str | BaseStorage | None = None,
        study_name: str | None = None,
        space: Mapping[str, Any] | None = None,
    ) -> TuneMemoryEstimate:
        """The memory envelope over the round ``tune`` with the same arguments
        would run next.

        The round is planned by the same function as ``tune`` (its effective
        ranges include an inherited range, a boundary expansion and an open
        round's recorded range). Each candidate point -- both ends of an
        ``"increasing"`` numeric dimension (stepped ones through the objective's
        grid), the lower end of the others, every choice of a categorical one --
        is assembled and assessed the way a trial is, and the largest estimate
        is the ``envelope``.

        ``status == "complete"`` promises only: while the study does not change,
        ``envelope.estimate_bytes`` is at least the estimate of every ordinary
        CPU trial sampled from the next round's effective ranges. It does not
        say which point the sampler draws, and it does not cover WAITING
        trials, values outside the ranges or a non-CPU device. The result is
        ``"incomplete"`` when the study holds a WAITING trial the next round
        would run, when a numeric dimension is outside the provider's closed
        set, when a candidate point cannot be evaluated (``failures``), or when
        the device may not be the CPU (``unresolved`` holds ``device_type``).
        An ``"incomplete"`` envelope is the maximum over the evaluated points,
        not an upper bound, and must not clear a whole-job check. A categorical
        dimension enumerates every choice and counts as resolved whatever its
        memory role; ``unresolved`` names numeric dimensions only (and the
        device). ``n_candidates`` counts the points that were evaluated, not the
        ones in ``failures``.

        The estimate covers the next round only: a boundary expansion after it
        is not estimated (when ``expand_boundary`` is on, a later round's range
        may be larger). It is an estimate of extra CPU memory, not a bound on
        the process's resident set. When the round runs no trial (an open round
        with no trial left), the result is ``complete`` with ``envelope=None``
        and ``n_candidates=0``.

        Nothing is written to the study, its trials or their attributes, and the
        Model is unchanged. The storage is opened as ``tune`` opens it, and
        opening an SQLite URL can create the file and an empty schema (the same
        side effect ``read_study_lineage`` has); do not call it when the storage
        file should not exist. Arguments, call matrix and refusals are those of
        ``tune``: ``resume=True`` with neither a held study nor a storage is
        ``TUNING_FAILED``, and a storage without the study plans round 1.

        Raises:
            LizyMLError: ``CONFIG_INVALID`` (``reason:
                "memory_estimate_unsupported"``) when the provider cannot
                estimate; whatever ``tune`` refuses before it writes.
        """
        cfg = self._cfg
        provider = require_memory_provider(get_provider(cfg.model))
        # The data step resets two per-call transients; put them back.
        block_values = self._block_values
        input_categories = self._input_categories
        try:
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
        finally:
            self._block_values = block_values
            self._input_categories = input_categories
        plan = ctx.plan
        if not plan.runs_optimize:
            return TuneMemoryEstimate(
                status="complete",
                envelope=None,
                unresolved=(),
                waiting_trials=0,
                round=plan.round_number,
                n_candidates=0,
                failures=(),
            )

        study = ctx.target.study
        waiting = (
            sum(1 for t in study.trials if t.state.name == "WAITING")
            if study is not None
            else 0
        )
        off_cpu = device_may_leave_cpu(
            provider,
            plan.effective,
            plan.fixed,
            ctx.base_model_params,
        )

        def build(
            model_p: dict[str, Any], smart_p: dict[str, Any], training_p: dict[str, Any]
        ) -> TrainComponents:
            return self._build_train_components(
                ctx.X,
                ctx.y,
                provider=provider,
                model_params=model_p,
                smart_params=smart_p,
                training_overrides=training_p,
            )

        found = evaluate_envelope(
            provider,
            plan.effective,
            base_model_params=ctx.base_model_params,
            base_smart_params=ctx.base_smart_params,
            fixed=dict(plan.fixed),
            # A trial keeps its outer folds alive together (H-0140).
            n_boosters=get_outer_n_splits(cfg),
            build_components=build,
            X=ctx.X,
        )
        unresolved = (("device_type",) if off_cpu else ()) + found.unresolved
        complete = not unresolved and not found.failures and waiting == 0
        return TuneMemoryEstimate(
            status="complete" if complete else "incomplete",
            envelope=found.envelope,
            unresolved=unresolved,
            waiting_trials=waiting,
            round=plan.round_number,
            n_candidates=found.n_candidates,
            failures=found.failures,
        )
