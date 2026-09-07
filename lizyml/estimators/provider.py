"""EstimatorProvider — protocol for multi-algorithm extensibility (H-0053).

Each estimator subpackage (e.g. ``lgbm/``) implements this protocol so that
``model.py`` can build TrainComponents without any estimator-specific imports.

See BLUEPRINT §14.4 for the full specification.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

from lizyml.core.types.search_dim import SearchDim
from lizyml.core.types.task import TaskType
from lizyml.estimators.base import BaseEstimatorAdapter
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
        feval_metadata: User-specified ``feval`` metric descriptors needed
            by the generated train.py to recompute custom metrics. Each
            dict has ``name``, ``params``, ``greater_is_better``,
            ``needs_proba``.
    """

    params: dict[str, Any]
    num_boost_round: int
    feval_metadata: list[dict[str, Any]] = field(default_factory=list)


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

    def build_pipeline_factory(self) -> Callable[[], BaseFeaturePipeline]:
        """Return a zero-arg factory that creates the appropriate FeaturePipeline."""
        ...

    def default_space(self, task: TaskType) -> list[SearchDim]:
        """Return the default hyperparameter search space for this estimator."""
        ...

    def default_fixed_params(self, task: TaskType) -> dict[str, Any]:
        """Return fixed params applied to every trial when using default space."""
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
