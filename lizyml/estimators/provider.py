"""EstimatorProvider — protocol for multi-algorithm extensibility (H-0053).

Each estimator subpackage (e.g. ``lgbm/``) implements this protocol so that
``model.py`` can build TrainComponents without any estimator-specific imports.

See BLUEPRINT §14.4 for the full specification.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd

from lizyml.core.types.search_dim import CategoricalDim, SearchDim
from lizyml.core.types.task import TaskType
from lizyml.estimators.base import BaseEstimatorAdapter
from lizyml.features.encoders.categorical_encoder import UnseenPolicy
from lizyml.features.pipeline_base import BaseFeaturePipeline

# H-0079: forward-typed alias for ``metric_choices`` return value. The
# ``Literal`` keys reserve room for future estimators that introduce
# additional metric sources (sklearn-backed metrics, custom Python
# callables, etc.) without breaking existing consumers.
MetricChoices = dict[Literal["native", "feval"], tuple[str, ...]]


@dataclass(frozen=True)
class ExportParams:
    """Codegen-relevant params extracted from a fitted estimator (H-0073).

    Returned by :meth:`EstimatorProvider.build_export_params` so that
    ``ModelPersistenceMixin.export_code()`` does not have to reach into
    estimator-specific private methods.

    Attributes:
        params: Native model parameters (e.g. LightGBM Booster API names).
        num_boost_round: Total training iterations actually used.
        early_stopping_rounds: The patience the fitted estimator actually
            trained with, or ``None`` when early stopping was off. Read from
            the trained adapter, not recomputed from config plus the current
            tuning result: ``tune()`` replaces the tuning result without
            replacing the fitted adapters, so the fitted estimator is the only
            surface that still answers what *this* model was trained with
            (H-0094 decision 13, review round 16).
            Deliberately carries **no default** -- a defaulted ``None`` would
            make "the provider did not set it" indistinguishable from "early
            stopping was off", which is the DC1 shape.
        feval_metadata: User-specified ``feval`` metric descriptors needed
            by the generated train.py to recompute custom metrics. Each
            dict has ``name``, ``params``, ``greater_is_better``,
            ``needs_proba``.
    """

    params: dict[str, Any]
    num_boost_round: int
    early_stopping_rounds: int | None
    feval_metadata: list[dict[str, Any]] = field(default_factory=list)


#: Refusal reasons of :class:`SmartValueRule` (H-0121 amendment 2, policy 1).
SMART_VALUE_INVALID = "smart_value_invalid"
SMART_VALUE_UNSUPPORTED_TASK = "smart_value_unsupported_task"

SmartValueKind = Literal["bool", "number", "untunable"]


@dataclass(frozen=True)
class SmartValueRule:
    """The values one smart parameter can hold (H-0121 amendment 2, policy 1).

    A ``category: smart`` dimension's sampled value becomes the trial's smart
    parameter as it is, and smart resolution reads it by truthiness and
    ``None``; so every value a dimension can take, and every value a stored
    tune result records, is held to this rule. Types are judged exactly:
    ``type(v) is bool`` for a bool, ``type(v) is int`` or ``type(v) is float``
    for a number (``bool``, numpy scalars and ``Decimal`` are not numbers).

    Attributes:
        kind: ``"bool"`` -- only ``True`` / ``False``, and only in a
            categorical dimension; ``"number"`` -- an ``int`` or ``float`` in
            the interval below, in a numeric or categorical dimension;
            ``"untunable"`` -- no search dimension and no recorded value.
        none_allowed: ``None`` is also a value (in a categorical dimension;
            a numeric dimension's bounds are never ``None``).
        low: A number's lower bound, exclusive.
        high: A number's upper bound, inclusive when ``high_inclusive``.
        high_inclusive: Whether ``high`` itself is a value. NaN and the
            infinities fall outside every interval.
        true_refused_tasks: Tasks for which ``True`` is refused with
            ``smart_value_unsupported_task``.
    """

    kind: SmartValueKind
    none_allowed: bool = False
    low: float = 0.0
    high: float = 1.0
    high_inclusive: bool = True
    true_refused_tasks: frozenset[str] = frozenset()

    def check_value(self, value: Any, task: TaskType) -> str | None:
        """Return the refusal reason for *value*, or ``None`` when it is held."""
        if self.kind == "untunable":
            return SMART_VALUE_INVALID
        if value is None:
            return None if self.none_allowed else SMART_VALUE_INVALID
        if self.kind == "bool":
            if type(value) is not bool:
                return SMART_VALUE_INVALID
            if value and task in self.true_refused_tasks:
                return SMART_VALUE_UNSUPPORTED_TASK
            return None
        if type(value) is not int and type(value) is not float:
            return SMART_VALUE_INVALID
        upper_ok = value <= self.high if self.high_inclusive else value < self.high
        return None if value > self.low and upper_ok else SMART_VALUE_INVALID

    def check_dimension(
        self, dim: SearchDim, task: TaskType
    ) -> tuple[str, Any, str] | None:
        """The first value of *dim* outside the rule, or ``None``.

        Returns:
            ``(key, value, reason)``: ``key`` is ``"choices"`` for a
            categorical dimension and ``"low"`` / ``"high"`` for a numeric
            one. A numeric dimension on a parameter that takes no numbers is
            reported at its ``low``, the first value it would sample.
        """
        if isinstance(dim, CategoricalDim):
            for choice in dim.choices:
                reason = self.check_value(choice, task)
                if reason is not None:
                    return "choices", choice, reason
            return None
        if self.kind == "number":
            for key in ("low", "high"):
                bound = getattr(dim, key)
                reason = self.check_value(bound, task)
                if reason is not None:
                    return key, bound, reason
            return None
        # A bool parameter, and one that takes no dimension, has no numeric one.
        return "low", dim.low, SMART_VALUE_INVALID

    def describe(self, task: TaskType) -> str:
        """What the rule holds, for a refusal message."""
        if self.kind == "untunable":
            return "nothing: it cannot be a search dimension"
        if self.kind == "bool":
            held = "True or False"
            if task in self.true_refused_tasks:
                held = f"False; True is not supported for task '{task}'"
            shown = f"a bool ({held})"
            where = ", in a categorical dimension only"
        else:
            close = "]" if self.high_inclusive else ")"
            shown = f"an int or float in ({self.low:g}, {self.high:g}{close}"
            where = ""
        if self.none_allowed:
            shown += " or None"
        return shown + where


class EstimatorProvider(Protocol):  # pragma: no cover
    """Uniform interface for estimator-specific logic.

    The Facade (``model.py``) calls these methods instead of importing
    estimator internals directly.  Adding a new algorithm requires only:

    1. Create ``estimators/<name>/`` with a provider implementing this protocol.
    2. Register it in ``_model_factories.get_provider()``.
    """

    def extract_model_params(self, model_cfg: Any) -> dict[str, Any]:
        """Extract native model parameters from a pydantic Config object."""
        ...

    def extract_smart_params(self, model_cfg: Any) -> dict[str, Any]:
        """Extract smart parameter fields from a pydantic Config object."""
        ...

    def accepted_model_param_names(self) -> frozenset[str]:
        """Return every native parameter name this estimator accepts (H-0093).

        Used by the Facade to reject a ``model.params`` key, or a
        ``tuning.optuna.space`` dimension declared ``category: model``, that the
        estimator would silently discard. LightGBM drops an unknown key without
        raising, so an unchecked typo produces a run that looks successful and
        in which the parameter did nothing.

        Implementations must derive this from the library rather than list it,
        so that an upstream rename is caught instead of being papered over.

        Returns:
            The accepted names, including any aliases the library honours.
        """
        ...

    def smart_param_names(self) -> frozenset[str]:
        """Return the names of this estimator's smart parameters (H-0093).

        These are LizyML's own parameters, not the library's. They are declared
        separately from :meth:`accepted_model_param_names` so that a smart name
        written where a native one belongs gets a diagnostic naming the category
        it wants, rather than the generic "unknown parameter" message -- writing
        ``num_leaves_ratio`` under ``category: model`` is a category mistake,
        not a typo.

        Must agree with the keys :meth:`extract_smart_params` returns; deriving
        both from one declaration is the way to keep that true.
        """
        ...

    def smart_param_owners(self) -> dict[str, str]:
        """Return ``{smart parameter: owner}`` for conditionally read ones (H-0121).

        The owner is the smart parameter that must be true for the first to be
        read by :meth:`resolve_smart_params`. A search dimension on a smart
        parameter whose owner cannot be true in a tune changes nothing, so the
        tuning layer refuses it (written by the user) or drops it (from the
        default space). Smart parameters read unconditionally are omitted; a
        provider without such parameters returns ``{}``.
        """
        ...

    def smart_value_rules(self) -> Mapping[str, SmartValueRule]:
        """Return ``{smart parameter: the values it can hold}`` (H-0121 am. 2).

        The one declaration of what a ``category: smart`` search dimension may
        take and what a stored tune result may record: ``Model.tune`` refuses
        a dimension outside it before any study write, ``restore_tuning`` and
        ``Model.load`` refuse a recorded value outside it. Must have a rule
        for every name :meth:`smart_param_names` returns. A range the config
        also validates must be read from the same declaration as the config's
        check, not restated.
        """
        ...

    def smart_managed_param_names(
        self, smart: dict[str, Any], task: TaskType
    ) -> dict[str, tuple[str, str]]:
        """Return names an *active* smart parameter will overwrite (H-0094).

        :meth:`resolve_smart_params` runs after the parameter dict is merged and
        its result wins, so a native name it writes cannot be set by hand: the
        value is replaced without a word. The caller uses this to refuse such a
        name instead, on every parameter layer at ``fit`` / ``tune`` (H-0094,
        H-0121).

        **Contract (H-0121 amendment 1).** The result depends on each smart
        value only through whether it is ``None`` and its truthiness, never on
        its magnitude: two values that are both ``None``, both falsy and not
        ``None``, or both truthy give the same result. ``Model.tune`` relies on
        it: it checks the parameter layers and the ``category: model``
        dimensions against one representative value per state of each
        ``category: smart`` dimension, combined over all of them
        (``lizyml.core._smart_combinations``), instead of against every value a
        trial can sample. Activation may depend on several smart parameters
        together. ``tests/test_estimators/test_smart_managed_contract.py``
        executes the contract on the LightGBM provider.

        Args:
            smart: Smart parameter values, as :meth:`extract_smart_params`
                returns them.
            task: ML task type -- a smart parameter may write a native name for
                one task and something that is not a parameter for another.

        Returns:
            ``{accepted spelling: (canonical name, the smart parameter)}``,
            empty when no smart parameter is active. Every spelling the library
            accepts must be a key: a library that resolves aliases makes a
            literal-name check admit the same parameter under another name.
        """
        ...

    def canonical_param_names(self, names: Iterable[str]) -> dict[str, str]:
        """Map each name to the parameter it identifies (H-0094).

        A library that accepts aliases treats two spellings as one parameter,
        so any code that merges parameter layers by dictionary key is merging
        by spelling rather than by identity: a lower-priority layer spelling it
        canonically survives beside a higher-priority layer spelling it as an
        alias, and the library then picks one of them. The caller uses this to
        drop the losing spelling before the estimator ever sees it.

        Args:
            names: Names to resolve. A name the estimator does not define maps
                to itself -- refusing it is :meth:`accepted_model_param_names`
                work, and this method must not double as that gate.

        Returns:
            ``{name: canonical name}`` for every name given.
        """
        ...

    def resolve_smart_params(
        self,
        smart: dict[str, Any],
        effective_params: dict[str, Any],
        n_rows: int,
        feature_names: list[str],
        y: pd.Series,
        task: TaskType,
    ) -> tuple[dict[str, Any], npt.NDArray[np.float64] | None]:
        """Resolve smart parameters to native estimator parameters.

        Returns:
            Tuple of (resolved native params, optional sample_weight).
        """
        ...

    def build_ratio_resolver(
        self,
        smart: dict[str, Any],
    ) -> Callable[[int], dict[str, Any]] | None:
        """Build a per-fold ratio resolver from smart params.

        Returns ``None`` when no ratio params are present.
        """
        ...

    def build_estimator_factory(
        self,
        task: TaskType,
        params: dict[str, Any],
        n_classes: int | None,
        early_stopping_rounds: int | None,
        seed: int,
    ) -> Callable[[], BaseEstimatorAdapter]:
        """Return a zero-arg factory that creates a configured estimator."""
        ...

    def build_pipeline_factory(
        self, unseen_policy: UnseenPolicy = "mode"
    ) -> Callable[[], BaseFeaturePipeline]:
        """Return a zero-arg factory that creates the appropriate FeaturePipeline.

        Args:
            unseen_policy: ``features.unseen_policy`` (H-0104). The facade passes
                the configured value when it fits; a pipeline restored from a
                saved state takes the policy recorded in that state instead.
        """
        ...

    def default_space(self, task: TaskType) -> list[SearchDim]:
        """Return the default hyperparameter search space for this estimator."""
        ...

    def default_fixed_params(self, task: TaskType) -> dict[str, Any]:
        """Return fixed params applied to every trial when using default space."""
        ...

    def default_params(self, task: TaskType) -> dict[str, Any]:
        """Return the native parameters every estimator for *task* starts from.

        The provider-decided defaults the adapter lays under the user's
        parameters (H-0122 spec 4.3). ``tune`` records them in the study's
        objective identity (``config.provider_defaults``), so a default that
        changes between rounds -- for instance with a LizyML upgrade -- refuses
        the resume instead of mixing two objectives in one study.

        Optional at runtime: ``tune`` reads it with ``getattr``, and a provider
        written before H-0122 without it contributes an empty mapping, i.e.
        its defaults are not checked. Implement it to have them checked.
        """
        ...

    def runtime_deps(self) -> dict[str, str]:
        """Return algorithm-specific dependency names and their versions.

        Used by ``_build_run_meta`` to populate ``RunMeta.deps_versions``.
        Example: ``{"lightgbm": "4.5.0"}``.
        """
        ...

    def params_summary(
        self,
        model: BaseEstimatorAdapter,
        model_cfg: Any,
    ) -> list[dict[str, Any]]:
        """Return parameter rows for ``params_table()``.

        Each row is ``{"parameter": str, "value": Any}``.
        Should include smart params and resolved native params.
        Per-fold best iterations are added by ``_model_tables.py``.
        """
        ...

    def build_export_params(self, adapter: BaseEstimatorAdapter) -> ExportParams:
        """Build codegen-relevant export parameters from a fitted adapter (H-0073).

        Used by :class:`~lizyml.core._model_persistence.ModelPersistenceMixin`
        to populate the codegen pipeline without importing estimator-specific
        adapter classes. The returned :class:`ExportParams` carries native
        booster params, the number of boosting rounds, and any ``feval``
        metric metadata needed to regenerate custom metric callables in the
        emitted ``train.py``.

        Args:
            adapter: The fitted estimator to export. The provider is
                responsible for narrowing the adapter type internally.
        """
        ...

    def parameter_bounds(self, task: TaskType) -> dict[str, dict[str, float | int]]:
        """Return per-parameter meaningful bounds for this estimator (H-0078).

        Used by ``expand_dims`` to clamp boundary expansion and by
        downstream UIs (e.g. LizyStudio) to constrain user input.

        Returns:
            Dict mapping a parameter name to ``{"min": ..., "max": ...}``.
            Parameters not present in the dict are unbounded (re-tune
            expansion grows freely as before). An empty dict means the
            provider has no parameter-specific bounds to declare.
        """
        ...

    def objective_choices(self, task: TaskType) -> tuple[str, ...]:
        """Return canonical objective names valid for *task* (H-0079).

        Used by:
        - :func:`default_space` to construct ``CategoricalDim("objective", ...)``.
        - Downstream UIs (LizyStudio) to populate the "objective" picker.
        - ``_build_params()`` / config validation to reject task-incompatible
          values.

        The returned tuple lists **canonical** names only — no aliases.
        Order is deterministic and stable across calls so UIs can render
        a consistent display.

        An empty tuple means the provider exposes no objective choices for
        this task (the consumer should treat that as "objective is not
        tunable / not user-selectable" for this estimator).
        """
        ...

    def metric_choices(self, task: TaskType) -> MetricChoices:
        """Return per-task valid metrics, split by source (H-0079).

        The returned dict has two keys:

        - ``"native"``: metrics evaluated by the underlying booster (e.g.
          via ``params["metric"]`` for LightGBM).
        - ``"feval"``:  metrics implemented by LizyML and wired as feval
          callables. Slower per trial than native because they execute
          Python on each evaluation round.

        Both tuples list **canonical** names only (aliases such as
        LightGBM's ``l1`` / ``l2`` are still accepted at config-input time
        by the metric_bridge, but ``metric_choices`` returns the canonical
        form so UIs can render a single, consistent picker).

        Order is deterministic. Names must not be duplicated across the
        two keys for a single task.
        """
        ...


@runtime_checkable
class TrialErrorProvider(Protocol):  # pragma: no cover
    """Optional provider capability: estimator errors that end one tuning
    trial rather than the round (H-0128).

    Separate from :class:`EstimatorProvider` so a provider written before
    H-0128 still satisfies that Protocol statically. The ``Model.tune``
    facade calls :meth:`trial_errors` only when
    ``isinstance(provider, TrialErrorProvider)``, and passes ``()`` otherwise.
    """

    def trial_errors(self) -> tuple[type[Exception], ...]:
        """Exception types a trial records as FAIL (for example the estimator
        library's own error type)."""
        ...
