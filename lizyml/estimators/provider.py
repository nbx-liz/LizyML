"""EstimatorProvider — protocol for multi-algorithm extensibility (H-0053).

Each estimator subpackage (e.g. ``lgbm/``) implements this protocol so that
``model.py`` can build TrainComponents without any estimator-specific imports.

See BLUEPRINT §14.4 for the full specification.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

from lizyml.core.types.search_dim import SearchDim
from lizyml.core.types.task import TaskType
from lizyml.estimators.base import BaseEstimatorAdapter
from lizyml.features.pipeline_base import BaseFeaturePipeline


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
