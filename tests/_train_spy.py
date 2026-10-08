"""One instrument for "what did LizyML actually hand LightGBM?".

Tests that assert on the *names and values* reaching the estimator cannot read
them from the config: the config is the input, and everything interesting
happens between it and ``lgb.train``. So they patch the library and record.

``lightgbm``, ``lizyml.estimators.lgbm.adapter.lgb`` and
``lizyml.calibration.isotonic.lgbm`` are the same module object, so patching
``lgb.train`` here observes the model's Boosters and the calibrator's alike.
``tests/test_calibration/test_calibration_param_names.py`` keeps its own
``_TrainSpy`` deliberately: it patches through ``isotonic.lgbm`` by name, which
is what makes it evidence about the *calibrator's* route rather than about
LightGBM in general.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import lightgbm as lgb


@contextlib.contextmanager
def record_lightgbm_calls() -> Iterator[dict[str, list[Any]]]:
    """Capture every ``lgb.train`` params dict and ``lgb.Dataset`` keyword.

    Patches the module attributes the adapter resolves at call time
    (``lizyml/estimators/lgbm/adapter.py`` holds ``import lightgbm as lgb`` and
    calls ``lgb.train`` / ``lgb.Dataset``), so the real functions still run and
    the recorded params are the ones a real Booster was trained on.

    Yields:
        ``{"train_params": [...], "dataset_kwargs": [...]}``, appended to in
        call order.
    """
    seen: dict[str, list[Any]] = {"train_params": [], "dataset_kwargs": []}
    real_train = lgb.train
    real_dataset = lgb.Dataset

    def spy_train(params: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        seen["train_params"].append(dict(params))
        return real_train(params, *args, **kwargs)

    def spy_dataset(*args: Any, **kwargs: Any) -> Any:
        seen["dataset_kwargs"].append(sorted(kwargs))
        return real_dataset(*args, **kwargs)

    lgb.train = spy_train  # type: ignore[assignment]
    lgb.Dataset = spy_dataset  # type: ignore[assignment,misc]
    try:
        yield seen
    finally:
        lgb.train = real_train  # type: ignore[assignment]
        lgb.Dataset = real_dataset  # type: ignore[assignment,misc]
