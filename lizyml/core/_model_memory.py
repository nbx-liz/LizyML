"""_model_memory -- the public memory estimates of ``Model`` (H-0150).

``Model.estimate_memory`` answers what one ``Model.fit`` with the same
arguments will ask the estimator library to allocate, without training and
without changing the Model. It shares fit's assessment-input step
(:meth:`ModelMemoryMixin._assessment_inputs`) and the provider's assessment
(``MemoryEstimateProvider.estimate_memory``, the one ``resource_warning``
reads) instead of copying either, so the number and the warning cannot differ
(DC3).

The estimate is of the extra CPU memory training adds, not a bound on the
process's resident set (H-0140): it leaves out isotonic calibration boosters
and the process's own base memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from lizyml.core._feature_name_check import check_feature_names_for_training
from lizyml.core._model_factories import (
    check_calibration_param_names,
    get_outer_n_splits,
    get_provider,
)
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.train_components import TrainComponents
from lizyml.core.types.memory_estimate import MemoryEstimate
from lizyml.estimators.provider import MemoryEstimateProvider

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

    from lizyml.config.schema import LizyMLConfig
    from lizyml.data.dataframe_builder import DataFrameComponents


def require_memory_provider(provider: Any) -> MemoryEstimateProvider:
    """*provider* when it has the memory-estimate capability, else
    ``CONFIG_INVALID`` (``reason: "memory_estimate_unsupported"``).

    Never answers ``None`` or 0 for a provider that cannot estimate (DC1).
    """
    if not isinstance(provider, MemoryEstimateProvider):
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                "This model's estimator provider cannot estimate memory; "
                "estimate_memory needs the MemoryEstimateProvider capability."
            ),
            context={
                "reason": "memory_estimate_unsupported",
                "provider": type(provider).__name__,
            },
        )
    return provider


class ModelMemoryMixin:
    """``Model.estimate_memory`` and fit's shared assessment-input step."""

    if TYPE_CHECKING:
        # --- Model body state this mixin reads (and puts back) ---
        _cfg: LizyMLConfig
        _tuning_result: Any
        _block_values: npt.NDArray[Any] | None
        _input_categories: dict[str, list[Any]]

        # --- Facade methods this mixin delegates to (defined on Model) ---
        def _merge_params(
            self,
            provider: Any,
            override: dict[str, Any] | None = None,
            *,
            validate_values: bool = False,
        ) -> tuple[dict[str, Any], dict[str, Any]]: ...

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

        def _prepare_training_data(
            self, data: pd.DataFrame | None
        ) -> tuple[
            pd.DataFrame, pd.Series, npt.NDArray[Any] | None, DataFrameComponents
        ]: ...

    def _assessment_inputs(
        self,
        provider: Any,
        X: pd.DataFrame,
        y: pd.Series,
        params: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], TrainComponents, dict[str, Any]]:
        """The smart parameters and train components a fit assesses.

        The step ``Model.fit`` and ``Model.estimate_memory`` both run between
        data preparation and the memory assessment: merge ``params`` over the
        config and the tuning result, refuse an unusable calibration name, and
        build the train components. Returns ``(smart_params, components,
        training_overrides)``.
        """
        cfg = self._cfg
        model_params, smart_params = self._merge_params(
            provider, override=params, validate_values=True
        )
        # Both parameter surfaces are checked here, before any training starts.
        # `_merge_params` gates `model.params`; the calibration surface is
        # checked beside it rather than at `_run_calibration`, which runs after
        # the whole outer CV (H-0093).
        check_calibration_param_names(cfg.calibration)
        training_overrides = (
            self._tuning_result.best_training_params
            if self._tuning_result is not None
            else {}
        )
        tc: TrainComponents = self._build_train_components(
            X,
            y,
            provider=provider,
            model_params=model_params,
            smart_params=smart_params,
            training_overrides=training_overrides,
        )
        return smart_params, tc, training_overrides

    def _estimate_inputs(
        self, data: pd.DataFrame | None
    ) -> tuple[pd.DataFrame, pd.Series, DataFrameComponents]:
        """Prepare the training data as fit does, leaving the Model unchanged.

        ``_prepare_training_data`` resets two per-call transients; they are put
        back so an estimate never alters what a later fit starts from.
        """
        block_values = self._block_values
        input_categories = self._input_categories
        try:
            X, y, _, components = self._prepare_training_data(data)
        finally:
            self._block_values = block_values
            self._input_categories = input_categories
        return X, y, components

    def estimate_memory(
        self,
        data: pd.DataFrame | None = None,
        params: dict[str, Any] | None = None,
    ) -> MemoryEstimate:
        """The extra CPU memory ``fit(data=data, params=params)`` would need.

        The same assessment ``fit`` computes before it trains: the same
        finalized parameters (aliases, ``num_class``, ``linear_tree``,
        ``auto_num_leaves``), the same feature bins, and ``outer folds + 1``
        boosters. Nothing is trained (no ``lightgbm.train`` or ``Dataset``
        call) and no Model state changes.

        This is an estimate of the extra CPU memory training adds, not a bound
        on the process's resident set: it leaves out isotonic calibration
        boosters and the process's base memory. A GPU run is not covered.

        Args:
            data: Training DataFrame, as for ``fit``.
            params: Parameter overrides, as for ``fit``.

        Returns:
            A :class:`~lizyml.core.types.memory_estimate.MemoryEstimate`.

        Raises:
            LizyMLError: ``CONFIG_INVALID`` with ``reason:
                "memory_estimate_unsupported"`` when the estimator provider
                cannot estimate; whatever ``fit`` refuses before training.
        """
        cfg = self._cfg
        provider = require_memory_provider(get_provider(cfg.model))
        X, y, _ = self._estimate_inputs(data)
        check_feature_names_for_training(provider, X)
        smart_params, tc, _ = self._assessment_inputs(provider, X, y, params)
        return provider.estimate_memory(
            estimator_factory=tc.estimator_factory,
            X=X,
            smart_params=smart_params,
            n_boosters=get_outer_n_splits(cfg) + 1,
        )
