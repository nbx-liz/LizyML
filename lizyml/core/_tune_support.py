"""_tune_support -- small helpers of ``Model.tune`` (moved out of
``_model_tuning`` to keep it under the file cap, H-0150)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from lizyml.core._model_metrics import _DEFAULT_METRICS
from lizyml.core.train_components import TrainComponents
from lizyml.estimators.provider import TrialErrorProvider
from lizyml.metrics.registry import parse_metric_entry

if TYPE_CHECKING:
    import pandas as pd

    from lizyml.config.schema import LizyMLConfig


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


def components_builder(
    model: Any, X: pd.DataFrame, y: pd.Series, provider: Any
) -> Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], TrainComponents]:
    """``(model_params, smart_params, training_overrides) -> TrainComponents``
    through *model*'s ``_build_train_components`` on *X*, *y* (H-0150)."""

    def build(
        model_params: dict[str, Any],
        smart_params: dict[str, Any],
        training_overrides: dict[str, Any],
    ) -> TrainComponents:
        tc: TrainComponents = model._build_train_components(
            X,
            y,
            provider=provider,
            model_params=model_params,
            smart_params=smart_params,
            training_overrides=training_overrides,
        )
        return tc

    return build


def _trial_errors(provider: Any) -> tuple[type[Exception], ...]:
    """The provider's ``trial_errors()`` when it has the optional
    :class:`TrialErrorProvider` capability, else none (H-0128)."""
    if isinstance(provider, TrialErrorProvider):
        return provider.trial_errors()
    return ()
