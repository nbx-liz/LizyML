"""LGBMAdapter — LightGBM estimator adapter for regression and classification."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

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

# Maps task → (objective, eval_metric)
_TASK_OBJECTIVE: dict[str, str] = {
    "regression": "regression",
    "binary": "binary",
    "multiclass": "multiclass",
}

_TASK_METRIC: dict[str, str] = {
    "regression": "rmse",
    "binary": "binary_logloss",
    "multiclass": "multi_logloss",
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

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw predictions."""
        model = self._require_fitted()
        result: np.ndarray = model.predict(X)
        return result

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
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
        result: np.ndarray = clf.predict_proba(X)
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
