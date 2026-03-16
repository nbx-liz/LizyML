"""EstimatorProvider — protocol for multi-algorithm extensibility (H-0053).

Each estimator subpackage (e.g. ``lgbm/``) implements this protocol so that
``model.py`` can build TrainComponents without any estimator-specific imports.

See BLUEPRINT §14.4 for the full specification.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

from lizyml.estimators.base import BaseEstimatorAdapter
from lizyml.features.pipeline_base import BaseFeaturePipeline
from lizyml.tuning.search_space import SearchDim


class EstimatorProvider(Protocol):
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
        task: str,
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
        task: str,
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

    def default_space(self, task: str) -> list[SearchDim]:
        """Return the default hyperparameter search space for this estimator."""
        ...

    def default_fixed_params(self, task: str) -> dict[str, Any]:
        """Return fixed params applied to every trial when using default space."""
        ...
