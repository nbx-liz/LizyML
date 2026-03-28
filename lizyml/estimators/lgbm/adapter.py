"""LGBMAdapter — LightGBM estimator adapter for regression and classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.estimators.base import BaseEstimatorAdapter, ImportanceKind
from lizyml.estimators.lgbm.defaults import (
    _COMMON_DEFAULTS,
    _TASK_METRIC,
    _TASK_OBJECTIVE,
)

try:
    import lightgbm as lgb
except ImportError as e:  # pragma: no cover
    raise LizyMLError(
        code=ErrorCode.OPTIONAL_DEP_MISSING,
        user_message="LightGBM is required. Install with: pip install lightgbm>=4.0",
        context={"package": "lightgbm"},
    ) from e

TaskType = Literal["regression", "binary", "multiclass"]


class LGBMAdapter(BaseEstimatorAdapter):
    """LightGBM adapter using the Booster API (``lgb.train``).

    Uses the native Booster API instead of the sklearn wrapper to avoid
    an intermittent ``model_to_string()`` bug (microsoft/LightGBM#7186).

    Args:
        task: ML task type.
        params: LightGBM parameters (excluding ``objective`` which is set
            automatically from *task*). ``metric`` may be user-specified;
            if absent or empty, falls back to task defaults (H-0061).
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
        self._categorical_features: list[str] | None = None

    def set_categorical_features(self, cols: list[str] | None) -> None:
        """Store categorical column names for use in ``fit()``."""
        self._categorical_features = cols

    def update_params(self, params: dict[str, Any]) -> None:
        """Update params before fit(). Used for per-fold ratio resolution."""
        self.params = {**self.params, **params}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        **kwargs: Any,
    ) -> LGBMAdapter:
        """Fit the LightGBM model via Booster API.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_valid: Optional validation features for early stopping.
            y_valid: Optional validation target for early stopping.
            **kwargs: Additional keyword arguments. ``sample_weight`` is
                extracted and passed to ``lgb.Dataset(weight=...)``.
        """
        self._feature_names = list(X_train.columns)
        params, num_boost_round = self._build_params()

        cat_feature: list[str] | Literal["auto"] = self._categorical_features or "auto"
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

        user_metric = params.get("metric")
        try:
            self._model = lgb.train(
                params,
                train_set,
                num_boost_round=num_boost_round,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks,
                keep_training_booster=True,
            )
        except lgb.basic.LightGBMError as exc:
            if "metric" in str(exc).lower():
                raise LizyMLError(
                    code=ErrorCode.CONFIG_INVALID,
                    user_message=(
                        f"Invalid LightGBM metric: {user_metric}. "
                        f"Check the metric name against LightGBM "
                        f"documentation. Original error: {exc}"
                    ),
                    context={
                        "metric": user_metric,
                        "task": self.task,
                    },
                ) from exc
            raise
        except ValueError as exc:
            if "eval metric" in str(exc).lower():
                raise LizyMLError(
                    code=ErrorCode.CONFIG_INVALID,
                    user_message=(
                        f"No valid eval metric for LightGBM. "
                        f"Specified metric={user_metric} may be "
                        f"invalid. Original error: {exc}"
                    ),
                    context={
                        "metric": user_metric,
                        "task": self.task,
                    },
                ) from exc
            raise

        # Detect silent invalid metric: LightGBM ignores unknown metric
        # names and produces empty eval_results when no valid metric
        # matched. Only check when user specified a custom metric.
        if (
            user_metric is not None
            and valid_sets is not None
            and not self._eval_results
        ):
            import warnings

            warnings.warn(
                f"LightGBM produced no eval results for "
                f"metric={user_metric}. The metric name(s) may be "
                f"invalid or unrecognized by this LightGBM version.",
                UserWarning,
                stacklevel=2,
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
        # Strip task-locked keys — objective is always set from task
        user_params.pop("objective", None)
        # Allow user-specified metric; fall back to task default if absent/empty
        user_metric = user_params.pop("metric", None)
        if user_metric:
            if isinstance(user_metric, str):
                user_metric = [user_metric]
            # Filter out empty strings
            user_metric = [m for m in user_metric if m]
            if user_metric:
                params["metric"] = user_metric
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

    def save_model_text(self, path: str | Path) -> Path:
        """Save the Booster to a human-readable text file.

        Args:
            path: Destination file path.

        Returns:
            The resolved Path.
        """
        booster = self._require_fitted()
        p = Path(path)
        booster.save_model(str(p))
        return p
