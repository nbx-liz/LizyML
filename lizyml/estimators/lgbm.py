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
    """LightGBM adapter using the Booster API (``lgb.train``).

    Uses the native Booster API instead of the sklearn wrapper to avoid
    an intermittent ``model_to_string()`` bug (microsoft/LightGBM#7186).

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

        self._model: lgb.Booster | None = None
        self._best_iteration: int | None = None
        self._feature_names: list[str] = []
        self._eval_results: dict[str, Any] = {}

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
        """Fit the LightGBM model via Booster API.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_valid: Optional validation features for early stopping.
            y_valid: Optional validation target for early stopping.
            categorical_feature: List of categorical column names.
            **kwargs: Additional keyword arguments. ``sample_weight`` is
                extracted and passed to ``lgb.Dataset(weight=...)``.
        """
        self._feature_names = list(X_train.columns)
        params, num_boost_round = self._build_params()

        cat_feature: list[str] | Literal["auto"] = categorical_feature or "auto"
        sample_weight = kwargs.pop("sample_weight", None)

        train_set = lgb.Dataset(
            X_train,
            label=y_train,
            weight=sample_weight,
            categorical_feature=cat_feature,
            free_raw_data=False,
        )

        callbacks: list[Any] = []
        valid_sets: list[lgb.Dataset] | None = None
        valid_names: list[str] | None = None

        if self.verbose_eval == -1:
            callbacks.append(lgb.log_evaluation(period=-1))
        elif self.verbose_eval > 0:
            callbacks.append(lgb.log_evaluation(period=self.verbose_eval))

        if X_valid is not None and y_valid is not None:
            valid_set = lgb.Dataset(
                X_valid,
                label=y_valid,
                reference=train_set,
                categorical_feature=cat_feature,
                free_raw_data=False,
            )
            valid_sets = [valid_set]
            valid_names = ["valid_0"]

            if self.early_stopping_rounds is not None:
                callbacks.append(
                    lgb.early_stopping(
                        stopping_rounds=self.early_stopping_rounds,
                        verbose=False,
                    )
                )

        self._eval_results = {}
        callbacks.append(lgb.record_evaluation(self._eval_results))

        self._model = lgb.train(
            params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
            keep_training_booster=True,
        )

        if self._model.best_iteration > 0:
            self._best_iteration = self._model.best_iteration

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Return predictions (regression values or class labels)."""
        booster = self._require_fitted()
        if self.task == "regression":
            raw = booster.predict(X)
            result: npt.NDArray[np.float64] = np.asarray(raw, dtype=np.float64)
            return result
        raw_proba = booster.predict(X)
        proba: npt.NDArray[np.float64] = np.asarray(raw_proba, dtype=np.float64)
        if self.task == "binary":
            labels: npt.NDArray[np.float64] = (proba > 0.5).astype(np.float64)
            return labels
        labels_mc: npt.NDArray[np.float64] = np.argmax(proba, axis=1).astype(np.float64)
        return labels_mc

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
        booster = self._require_fitted()
        raw = booster.predict(X)
        proba: npt.NDArray[np.float64] = np.asarray(raw, dtype=np.float64)
        if self.task == "binary":
            result: npt.NDArray[np.float64] = np.column_stack([1.0 - proba, proba])
            return result
        # multiclass: already (n, k)
        return proba

    def predict_raw(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Return raw scores (logits) before sigmoid/softmax.

        For regression, identical to ``predict()``.
        For binary/multiclass, returns booster raw_score output.
        """
        if self.task == "regression":
            return self.predict(X)
        booster = self._require_fitted()
        raw = booster.predict(X, raw_score=True)
        result: npt.NDArray[np.float64] = np.asarray(raw, dtype=np.float64)
        return result

    # ------------------------------------------------------------------
    # Importance
    # ------------------------------------------------------------------

    def importance(self, kind: ImportanceKind = "split") -> dict[str, float]:
        """Return feature importance scores.

        Args:
            kind: ``"split"`` or ``"gain"``.
        """
        booster = self._require_fitted()
        importance_type = "split" if kind == "split" else "gain"
        values = booster.feature_importance(importance_type=importance_type)
        return {
            name: float(val)
            for name, val in zip(self._feature_names, values, strict=True)
        }

    # ------------------------------------------------------------------
    # Native model
    # ------------------------------------------------------------------

    def get_native_model(self) -> lgb.Booster:
        return self._require_fitted()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def best_iteration(self) -> int | None:
        return self._best_iteration

    @property
    def eval_results(self) -> dict[str, Any]:
        """Evaluation results collected during training via ``record_evaluation``.

        Structure: ``{"valid_0": {"metric_name": [val_per_iter, ...]}}``.
        Empty dict when no validation set was used.
        """
        return self._eval_results

    # ------------------------------------------------------------------
    # Serialization (backward compat with sklearn wrapper models)
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        # Old format may lack _eval_results
        if not hasattr(self, "_eval_results"):
            object.__setattr__(self, "_eval_results", {})
        # Migrate old sklearn wrapper (_model = LGBMRegressor/LGBMClassifier)
        model = self._model
        if model is not None and hasattr(model, "booster_"):
            self._model = model.booster_
            if hasattr(model, "best_iteration_") and model.best_iteration_ > 0:
                self._best_iteration = int(model.best_iteration_)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_params(self) -> tuple[dict[str, Any], int]:
        """Build LightGBM params dict and num_boost_round.

        Returns:
            ``(params_dict, num_boost_round)`` tuple.
            ``params_dict`` uses Booster API naming (``seed``, ``verbosity``).
            ``num_boost_round`` is extracted from ``n_estimators``.
        """
        params: dict[str, Any] = {
            "objective": _TASK_OBJECTIVE[self.task],
            "metric": _TASK_METRIC[self.task],
            **{k: v for k, v in _COMMON_DEFAULTS.items() if k != "n_estimators"},
            "seed": self.random_state,
            "verbosity": -1,
        }
        if self.task == "multiclass" and self.num_class is not None:
            params["num_class"] = self.num_class

        # Extract num_boost_round from user params (n_estimators) or use default
        user_params = dict(self.params)
        num_boost_round = int(
            user_params.pop("n_estimators", _COMMON_DEFAULTS["n_estimators"])
        )
        # Normalize sklearn param names → Booster API names
        if "random_state" in user_params:
            user_params.setdefault("seed", user_params.pop("random_state"))
        if "verbose" in user_params:
            user_params.setdefault("verbosity", user_params.pop("verbose"))
        params.update(user_params)

        return params, num_boost_round

    def _require_fitted(self) -> lgb.Booster:
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
