"""_resource_warning -- the pre-training memory warning of fit and tune (H-0140).

``Model.fit`` checks once, before its CV folds and refit train; ``Model.tune``
checks every trial and warns at the first one that fires in the call, then
logs the rest. The check is the provider's optional
:class:`~lizyml.estimators.provider.ResourceWarningProvider` capability: a
provider without it is not checked.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import pandas as pd

from lizyml.core.logging import get_logger
from lizyml.core.train_components import TrainComponents
from lizyml.estimators.provider import ResourceWarningProvider

_log = get_logger("core.resource_warning")


@dataclass
class WarningLatch:
    """One user warning per ``tune()`` call; later trials only log."""

    warned: bool = False


def check_resources(
    provider: Any,
    tc: TrainComponents,
    X: pd.DataFrame,
    smart_params: dict[str, Any],
    *,
    n_boosters: int,
    stacklevel: int,
    latch: WarningLatch | None = None,
) -> None:
    """Warn (``UserWarning``) before training when the provider says so.

    Args:
        provider: The fit's estimator provider.
        tc: The fit's train components (their estimator factory is read).
        X: The training features.
        smart_params: The fit's smart parameters.
        n_boosters: The trained models the fit keeps alive at once.
        stacklevel: Passed to ``warnings.warn`` so the warning points at the
            caller's ``fit`` / ``tune`` line.
        latch: For ``tune``: warn at the first firing trial of the call only.
    """
    if not isinstance(provider, ResourceWarningProvider):
        return
    text = provider.resource_warning(
        estimator_factory=tc.estimator_factory,
        X=X,
        smart_params=smart_params,
        n_boosters=n_boosters,
    )
    if text is None:
        return
    if latch is not None:
        if latch.warned:
            _log.info("event='tune.trial.memory_warning' %s", text)
            return
        latch.warned = True
    warnings.warn(text, UserWarning, stacklevel=stacklevel + 1)
