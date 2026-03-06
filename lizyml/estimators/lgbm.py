"""LGBMAdapter — LightGBM estimator adapter for regression and classification."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
    from lizyml.config.schema import LGBMConfig

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.registries import EstimatorRegistry
from lizyml.estimators.base import BaseEstimatorAdapter, ImportanceKind

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError as e:  # pragma: no cover
    raise LizyMLError(
        code=ErrorCode.OPTIONAL_DEP_MISSING,
        user_message="LightGBM is required. Install with: pip install lightgbm>=4.0",
        context={"package": "lightgbm"},
    ) from e

TaskType = Literal["regression", "binary", "multiclass"]

# Maps task → objective
_TASK_OBJECTIVE: dict[str, str] = {
    "regression": "huber",
    "binary": "binary",
    "multiclass": "multiclass",
}

# Maps task → eval_metric list
_TASK_METRIC: dict[str, list[str]] = {
    "regression": ["huber", "mae", "mape"],
    "binary": ["auc", "binary_logloss"],
    "multiclass": ["auc_mu", "multi_logloss"],
}

# Common LightGBM defaults (also used by resolve_smart_params)
_COMMON_DEFAULTS: dict[str, Any] = {
    "boosting": "gbdt",
    "n_estimators": 1500,
    "learning_rate": 0.001,
    "max_depth": 5,
    "max_bin": 511,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 10,
    "lambda_l1": 0.0,
    "lambda_l2": 0.000001,
    "first_metric_only": False,
}


@EstimatorRegistry.register("lgbm")
class LGBMAdapter(BaseEstimatorAdapter):
    """LightGBM adapter supporting regression, binary, and multiclass tasks.

    Args:
        task: ML task type.
        params: LightGBM parameters (excluding ``objective`` and ``metric``
            which are set automatically from *task*).
        num_class: Number of classes for multiclass (required when
            ``task="multiclass"``).
        early_stopping_rounds: Early stopping patience.
        verbose_eval: Evaluation verbose interval (``-1`` to suppress).
        random_state: Random seed.
    """

    def __init__(
        self,
        task: TaskType = "regression",
        params: dict[str, Any] | None = None,
        num_class: int | None = None,
        early_stopping_rounds: int | None = 50,
        verbose_eval: int = -1,
        random_state: int = 42,
    ) -> None:
        self.task = task
        self.params = params or {}
        self.num_class = num_class
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self.random_state = random_state

        self._model: LGBMRegressor | LGBMClassifier | None = None
        self._best_iteration: int | None = None
        self._feature_names: list[str] = []

    def update_params(self, params: dict[str, Any]) -> None:
        """Update params before fit(). Used for per-fold ratio resolution."""
        self.params.update(params)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        categorical_feature: list[str] | None = None,
        **kwargs: Any,
    ) -> LGBMAdapter:
        """Fit the LightGBM model.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_valid: Optional validation features for early stopping.
            y_valid: Optional validation target for early stopping.
            categorical_feature: List of categorical column names.
            **kwargs: Forwarded to the underlying LightGBM estimator's ``fit``.
        """
        self._feature_names = list(X_train.columns)
        base_params = self._build_params()
        base_params.update(self.params)

        callbacks: list[Any] = []
        if self.verbose_eval == -1:
            callbacks.append(lgb.log_evaluation(period=-1))
        elif self.verbose_eval > 0:
            callbacks.append(lgb.log_evaluation(period=self.verbose_eval))

        eval_set = None
        if X_valid is not None and y_valid is not None:
            eval_set = [(X_valid, y_valid)]
            if self.early_stopping_rounds is not None:
                callbacks.append(
                    lgb.early_stopping(
                        stopping_rounds=self.early_stopping_rounds,
                        verbose=False,
                    )
                )

        cat_feature = categorical_feature or "auto"

        if self.task == "regression":
            self._model = LGBMRegressor(**base_params)
        else:
            self._model = LGBMClassifier(**base_params)

        self._model.fit(
            X_train,
            y_train,
            eval_set=eval_set,  # type: ignore[arg-type]
            categorical_feature=cat_feature,  # type: ignore[arg-type]
            callbacks=callbacks,
            **kwargs,
        )

        if hasattr(self._model, "best_iteration_") and self._model.best_iteration_ > 0:
            self._best_iteration = int(self._model.best_iteration_)

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Return raw predictions."""
        model = self._require_fitted()
        result: npt.NDArray[np.float64] = model.predict(X)
        return result

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Return class probabilities.

        For binary tasks returns shape ``(n, 2)``.
        For multiclass returns shape ``(n, k)``.

        Raises:
            :class:`~lizyml.core.exceptions.LizyMLError` with
            ``UNSUPPORTED_TASK`` for regression.
        """
        if self.task == "regression":
            raise LizyMLError(
                code=ErrorCode.UNSUPPORTED_TASK,
                user_message="predict_proba is not available for regression tasks.",
                context={"task": self.task},
            )
        clf = self._require_fitted()
        if not isinstance(clf, LGBMClassifier):
            raise LizyMLError(  # pragma: no cover
                code=ErrorCode.UNSUPPORTED_TASK,
                user_message="predict_proba requires a classifier.",
                context={"task": self.task},
            )
        result: npt.NDArray[np.float64] = clf.predict_proba(X)
        return result

    def predict_raw(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Return raw scores (logits) before sigmoid/softmax.

        For regression, identical to ``predict()``.
        For binary/multiclass, returns booster raw_score output.
        """
        if self.task == "regression":
            return self.predict(X)
        model = self._require_fitted()
        result: npt.NDArray[np.float64] = model.booster_.predict(  # type: ignore[assignment]
            X, raw_score=True
        )
        return result

    # ------------------------------------------------------------------
    # Importance
    # ------------------------------------------------------------------

    def importance(self, kind: ImportanceKind = "split") -> dict[str, float]:
        """Return feature importance scores.

        Args:
            kind: ``"split"`` or ``"gain"``.
        """
        model = self._require_fitted()
        importance_type = "split" if kind == "split" else "gain"
        values = model.booster_.feature_importance(importance_type=importance_type)
        return {
            name: float(val)
            for name, val in zip(self._feature_names, values, strict=True)
        }

    # ------------------------------------------------------------------
    # Native model
    # ------------------------------------------------------------------

    def get_native_model(self) -> LGBMRegressor | LGBMClassifier:
        return self._require_fitted()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def best_iteration(self) -> int | None:
        return self._best_iteration

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "objective": _TASK_OBJECTIVE[self.task],
            "metric": _TASK_METRIC[self.task],
            **_COMMON_DEFAULTS,
            "random_state": self.random_state,
            "verbose": -1,
        }
        if self.task == "multiclass" and self.num_class is not None:
            params["num_class"] = self.num_class
        return params

    def _require_fitted(self) -> LGBMRegressor | LGBMClassifier:
        if self._model is None:
            raise LizyMLError(
                code=ErrorCode.MODEL_NOT_FIT,
                user_message="LGBMAdapter has not been fitted yet. Call fit() first.",
                context={"adapter": "LGBMAdapter"},
            )
        return self._model


# ------------------------------------------------------------------
# Smart parameter resolution (H-0021)
# ------------------------------------------------------------------


def _compute_num_leaves(max_depth: int | None, ratio: float) -> int:
    """Compute num_leaves from max_depth and ratio."""
    base = 131072 if max_depth is None or max_depth < 0 else 2**max_depth
    return max(8, min(131072, math.ceil(base * ratio)))


def _compute_ratio_param(n_rows: int, ratio: float) -> int:
    """Convert a ratio to an absolute count (min 1)."""
    return max(1, math.ceil(n_rows * ratio))


def resolve_smart_params(
    config: LGBMConfig,
    effective_params: dict[str, Any],
    n_rows: int,
    feature_names: list[str],
    y: pd.Series,
    task: TaskType,
) -> tuple[dict[str, Any], npt.NDArray[np.float64] | None]:
    """Resolve smart parameters to native LightGBM parameters.

    Args:
        config: LGBMConfig with smart parameter fields.
        effective_params: Merged params (defaults + user + best_params).
        n_rows: Number of training rows.
        feature_names: List of feature column names.
        y: Target series.
        task: ML task type.

    Returns:
        Tuple of (resolved native params dict, sample_weight array or None).
    """
    resolved: dict[str, Any] = {}
    sample_weight: npt.NDArray[np.float64] | None = None

    # auto_num_leaves
    if config.auto_num_leaves:
        resolved["num_leaves"] = _compute_num_leaves(
            effective_params.get("max_depth"), config.num_leaves_ratio
        )

    # NOTE: ratio params (min_data_in_leaf_ratio, min_data_in_bin_ratio) are
    # resolved per-fold via resolve_ratio_params() using inner_train size (H-0036).

    # feature_weights
    if config.feature_weights is not None:
        unknown = set(config.feature_weights) - set(feature_names)
        if unknown:
            raise LizyMLError(
                code=ErrorCode.CONFIG_INVALID,
                user_message=f"Unknown features in feature_weights: {sorted(unknown)}",
                context={"unknown_features": sorted(unknown)},
            )
        weights = [config.feature_weights.get(f, 1.0) for f in feature_names]
        resolved["feature_weights"] = weights
        resolved["feature_pre_filter"] = False

    # balanced — None means auto (True for binary/multiclass, False for regression)
    effective_balanced = config.balanced
    if effective_balanced is None:
        effective_balanced = task != "regression"
    if effective_balanced:
        if task == "regression":
            raise LizyMLError(
                code=ErrorCode.UNSUPPORTED_TASK,
                user_message="'balanced' is not supported for regression tasks.",
                context={"task": task},
            )
        if task == "binary":
            neg = int((y == 0).sum())
            pos = int((y == 1).sum())
            resolved["scale_pos_weight"] = neg / pos if pos > 0 else 1.0
        else:  # multiclass
            from sklearn.utils.class_weight import compute_sample_weight

            sw: npt.NDArray[np.float64] = compute_sample_weight("balanced", y)
            sample_weight = sw

    return resolved, sample_weight


def resolve_smart_params_from_dict(
    smart_params: dict[str, Any],
    effective_params: dict[str, Any],
    n_rows: int,
) -> dict[str, Any]:
    """Resolve smart parameters from a flat dict (for tuning trials).

    Supports ``num_leaves_ratio``, ``min_data_in_leaf_ratio``, and
    ``min_data_in_bin_ratio``.

    Args:
        smart_params: Dict with smart param names and values.
        effective_params: Merged model params (for max_depth lookup).
        n_rows: Number of training rows.

    Returns:
        Dict of resolved native LightGBM parameters.
    """
    resolved: dict[str, Any] = {}

    if "num_leaves_ratio" in smart_params:
        resolved["num_leaves"] = _compute_num_leaves(
            effective_params.get("max_depth"),
            smart_params["num_leaves_ratio"],
        )

    # NOTE: ratio params (min_data_in_leaf_ratio, min_data_in_bin_ratio) are
    # resolved per-fold via resolve_ratio_params() using inner_train size (H-0036).

    return resolved


def resolve_ratio_params(
    min_data_in_leaf_ratio: float | None,
    min_data_in_bin_ratio: float | None,
    n_rows: int,
) -> dict[str, int]:
    """Resolve n_rows-dependent ratio params to native LightGBM values.

    Called per-fold with inner_train size (after inner_valid split) to ensure
    ratio params reflect the actual training data size (H-0036).

    Args:
        min_data_in_leaf_ratio: Ratio for min_data_in_leaf (None to skip).
        min_data_in_bin_ratio: Ratio for min_data_in_bin (None to skip).
        n_rows: Number of inner-train rows (after inner_valid split).

    Returns:
        Dict of resolved native LightGBM parameters.
    """
    resolved: dict[str, int] = {}
    if min_data_in_leaf_ratio is not None:
        resolved["min_data_in_leaf"] = _compute_ratio_param(
            n_rows, min_data_in_leaf_ratio
        )
    if min_data_in_bin_ratio is not None:
        resolved["min_data_in_bin"] = _compute_ratio_param(
            n_rows, min_data_in_bin_ratio
        )
    return resolved
