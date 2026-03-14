"""IsotonicCalibrator — isotonic regression via LGBM monotone constraints.

Uses a single-feature LightGBM Booster with ``monotone_constraints=[1]``
to learn a monotone mapping from raw scores to calibrated probabilities
(BLUEPRINT §12.2, H-0047).
"""

from __future__ import annotations

import math
from typing import Any

import lightgbm as lgbm
import numpy as np
import numpy.typing as npt

from lizyml.calibration.base import BaseCalibratorAdapter
from lizyml.core.registries import CalibratorRegistry

_ISOTONIC_DEFAULTS: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "monotone_constraints": [1],
    "monotone_constraints_method": "advanced",
    "num_leaves": 7,
    "max_depth": 3,
    "min_data_in_leaf_ratio": 0.01,
    "learning_rate": 0.03,
    "lambda_l2": 5.0,
    "min_gain_to_split": 0.0,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
}

_NUM_BOOST_ROUND = 1000
_EARLY_STOPPING_ROUNDS = 100
_VALIDATION_RATIO = 0.1
_DEFAULT_SEED = 42
_MIN_SAMPLES_FOR_EARLY_STOPPING = 20


def _sigmoid(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Numerically stable sigmoid: avoids overflow for large |x|."""
    result: npt.NDArray[np.float64] = np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )
    return result


@CalibratorRegistry.register("isotonic")
class IsotonicCalibrator(BaseCalibratorAdapter):
    """Isotonic calibration using LGBM Booster with monotone constraints.

    Learns a monotone non-decreasing mapping from raw scores (logits)
    to calibrated probabilities using a single-feature LightGBM Booster.

    Args:
        params: Optional parameter overrides.  ``monotone_constraints``
            is always forced to ``[1]`` and cannot be overridden.
            Special keys (not passed to LightGBM):
            - ``num_boost_round``: number of boosting rounds (default 1000)
            - ``validation_ratio``: fraction for early stopping (default 0.1)
            - ``seed``: random seed for validation split (default 42)
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        # Copy to avoid mutating the caller's dict
        user = dict(params) if params else {}

        # Extract non-LightGBM params
        self._num_boost_round = int(user.pop("num_boost_round", _NUM_BOOST_ROUND))
        self._validation_ratio = float(user.pop("validation_ratio", _VALIDATION_RATIO))
        self._seed = int(user.pop("seed", _DEFAULT_SEED))

        if not (0.0 < self._validation_ratio < 1.0):
            from lizyml.core.exceptions import ErrorCode, LizyMLError

            raise LizyMLError(
                code=ErrorCode.CONFIG_INVALID,
                user_message="validation_ratio must be in (0.0, 1.0)",
                context={"validation_ratio": self._validation_ratio},
            )

        merged = {**_ISOTONIC_DEFAULTS, **user}
        # Always enforce monotone constraint
        merged["monotone_constraints"] = [1]
        merged["verbose"] = -1
        merged["seed"] = self._seed
        self._lgbm_params = merged
        self._model: lgbm.Booster | None = None

    @property
    def name(self) -> str:
        return "isotonic"

    def fit(
        self, oof_scores: npt.NDArray[np.float64], y: npt.NDArray[Any]
    ) -> IsotonicCalibrator:
        X_cal = oof_scores.reshape(-1, 1)
        y_float = y.astype(float)
        n_samples = len(X_cal)

        # Resolve min_data_in_leaf_ratio to absolute value
        params = dict(self._lgbm_params)
        ratio = params.pop("min_data_in_leaf_ratio", None)
        if ratio is not None:
            params["min_data_in_leaf"] = max(1, math.ceil(n_samples * ratio))

        use_early_stopping = n_samples >= _MIN_SAMPLES_FOR_EARLY_STOPPING

        if use_early_stopping:
            # Split for early stopping validation
            rng = np.random.default_rng(self._seed)
            n_valid = max(1, int(n_samples * self._validation_ratio))
            indices = rng.permutation(n_samples)
            valid_idx = indices[:n_valid]
            train_idx = indices[n_valid:]

            train_ds = lgbm.Dataset(X_cal[train_idx], label=y_float[train_idx])
            valid_ds = lgbm.Dataset(
                X_cal[valid_idx], label=y_float[valid_idx], reference=train_ds
            )

            self._model = lgbm.train(
                params,
                train_ds,
                num_boost_round=self._num_boost_round,
                valid_sets=[valid_ds],
                valid_names=["valid"],
                callbacks=[
                    lgbm.early_stopping(
                        stopping_rounds=_EARLY_STOPPING_ROUNDS, verbose=False
                    ),
                    lgbm.log_evaluation(period=0),
                ],
            )
        else:
            # Too few samples: no early stopping, train on all data
            train_ds = lgbm.Dataset(X_cal, label=y_float)
            self._model = lgbm.train(
                params,
                train_ds,
                num_boost_round=self._num_boost_round,
                callbacks=[lgbm.log_evaluation(period=0)],
            )

        return self

    def predict(self, scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self._model is None:
            raise RuntimeError("IsotonicCalibrator has not been fitted.")
        X = scores.reshape(-1, 1)
        # objective="binary" Booster returns raw scores (log-odds)
        raw_pred = self._model.predict(X)
        raw: npt.NDArray[np.float64] = np.asarray(raw_pred, dtype=np.float64)
        proba = _sigmoid(raw)
        clipped: npt.NDArray[np.float64] = np.clip(proba, 0.0, 1.0)
        return clipped
