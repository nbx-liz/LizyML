"""LGBMAdapter — LightGBM estimator adapter for regression and classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.param_domain import assert_plain_params
from lizyml.estimators.base import BaseEstimatorAdapter, ImportanceKind
from lizyml.estimators.lgbm.defaults import (
    _COMMON_DEFAULTS,
    _TASK_METRIC,
    _TASK_OBJECTIVE,
    TASK_COMPATIBLE_OBJECTIVES,
)
from lizyml.estimators.lgbm.metric_bridge import resolve_metrics
from lizyml.estimators.lgbm.param_names import (
    LGBM_CANONICAL_NAME,
    accepted_spellings,
)


def _pop_by_identity(
    user_params: dict[str, Any], canonical: str
) -> tuple[Any, str | None]:
    """Pop every spelling of one parameter, and return its single value.

    The special handling below (objective, metric, boosting rounds) used to
    match one literal name, so the same parameter written under an alias was
    left in the ordinary parameter dict: it skipped the validation the literal
    name gets, and -- once identity-aware merging removed the shadowing default
    -- became the value that trained (H-0094, review round 4). Measured on a
    binary task: ``fit(params={"application": "regression"})`` trained a
    regression objective.

    Args:
        user_params: Mutated in place; every spelling found is removed.
        canonical: The canonical parameter name.

    Returns:
        ``(value, the spelling it was written as)``, or ``(None, None)``.

    Raises:
        LizyMLError: with ``CONFIG_INVALID`` when one layer names the parameter
            under more than one spelling. **The values are not read** (H-0096);
            picking one silently is the class of defect this whole change exists
            to remove, and deciding whether two values are "the same" is the
            question that had no closed domain. Kept in step with
            ``check_duplicate_identities`` by asking the same thing -- how many
            spellings -- rather than by sharing a comparison.
    """
    supplied = {
        name: user_params.pop(name)
        for name in list(user_params)
        if name in accepted_spellings(canonical)
    }
    if not supplied:
        return None, None
    if len(supplied) > 1:
        # The message names the **spellings** and not the values. A spelling is
        # a `str` key and always prints; a value need not -- a Python `int`
        # above `sys.get_int_max_str_digits()` digits has no decimal text, and
        # formatting one here turned the promised `CONFIG_INVALID` into a bare
        # `ValueError` (review round 27). The rule decides on how many spellings
        # were written, so reporting it must not depend on the values either.
        # `check_duplicate_identities` carries the same correction; the defect
        # was reported at one of the two and was present at both.
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                f"'{canonical}' is set more than once, under the spellings "
                f"{sorted(supplied)}. LightGBM treats these as one parameter. "
                "Write it once, under one spelling."
            ),
            context={"parameter": canonical, "spellings": sorted(supplied)},
        )
    written, value = next(iter(supplied.items()))
    return value, written


def _check_objective_compatible(task: str, objective: str) -> None:
    """Raise CONFIG_INVALID when *objective* is not valid for *task* (H-0079).

    Cross-task injection (e.g. ``objective='regression'`` for binary task)
    used to be silently stripped pre-H-0079 — same defensive intent, but
    explicit failure instead of a silent override that misled tuning_table.
    """
    valid = TASK_COMPATIBLE_OBJECTIVES.get(task, frozenset())
    if objective not in valid:
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                f"objective '{objective}' is not compatible with task "
                f"'{task}'. Valid objectives: {sorted(valid)}."
            ),
            context={
                "task": task,
                "objective": objective,
                "valid_objectives": sorted(valid),
            },
        )


try:
    import lightgbm as lgb
except ImportError as e:  # pragma: no cover
    raise LizyMLError(
        code=ErrorCode.OPTIONAL_DEP_MISSING,
        user_message="LightGBM is required. Install with: pip install lightgbm>=4.0",
        context={"package": "lightgbm"},
    ) from e

from lizyml.core.types.task import TaskType  # noqa: E402  re-export for ext

__all__ = ["LGBMAdapter", "TaskType"]


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
        self._feval_display_names: list[str] = []

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
        params, num_boost_round, feval_list, self._feval_display_names = (
            self._build_params()
        )

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
        # H-0095: the domain is closed at the four surfaces, and this is
        # where that becomes a property rather than a claim about wiring.
        # A value that reached training without being normalised stops the
        # run and names itself, instead of being serialised by whatever
        # `__format__` it happens to carry.
        assert_plain_params(params, where="lgb.train")
        try:
            self._model = lgb.train(
                params,
                train_set,
                num_boost_round=num_boost_round,
                valid_sets=valid_sets,
                valid_names=valid_names,
                feval=feval_list if feval_list else None,
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
        # H-0065: Old format may lack _feval_display_names
        if not hasattr(self, "_feval_display_names"):
            object.__setattr__(self, "_feval_display_names", [])
        # Migrate old sklearn wrapper (_model = LGBMRegressor/LGBMClassifier)
        model = self._model
        if model is not None and hasattr(model, "booster_"):
            self._model = model.booster_
            if hasattr(model, "best_iteration_") and model.best_iteration_ > 0:
                self._best_iteration = int(model.best_iteration_)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_params(
        self,
    ) -> tuple[dict[str, Any], int, list[Any], list[str]]:
        """Build LightGBM params, num_boost_round, feval list, and names.

        Returns:
            ``(params_dict, num_boost_round, feval_list, feval_display_names)`` tuple.
            ``params_dict`` uses Booster API naming (``seed``, ``verbosity``).
            ``num_boost_round`` is extracted from ``n_estimators``.
            ``feval_list`` contains callables for LizyML-only metrics.
            ``feval_display_names`` contains human-readable names for feval
            metrics (e.g. ``"precision_at_k (k=20)"``).
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
        rounds_value, _ = _pop_by_identity(user_params, "num_iterations")
        num_boost_round = int(
            _COMMON_DEFAULTS["n_estimators"] if rounds_value is None else rounds_value
        )
        # Normalize sklearn param names → Booster API names, **by identity**.
        # These read `user_params` by one literal spelling each, so
        # `random_state` was renamed and `random_seed` -- the third spelling of
        # the same parameter -- was not.
        #
        # **This is a consistency fix, not a defect fix, and the difference was
        # executed rather than assumed.** `random_seed=7` reached `lgb.train`
        # under its own name and LightGBM honoured it: the booster is identical
        # to one trained with `seed=7` and differs from `seed=99`. So no value
        # was lost. What this removes is the last literal-spelling read of a
        # caller-owned dict in this module -- the construct that *did* cost a
        # defect one file over, where `_extract_feval_metadata` read `"metric"`
        # literally and dropped a custom metric from the export (decision 9).
        #
        # On the facade path the branch is unreachable anyway:
        # `check_training_managed_overrides` claims every spelling of `seed`
        # whenever `training.seed` is set, and it is always set (default 42; an
        # explicit null is refused). `LGBMAdapter` is also constructed directly,
        # which is the path this still governs.
        # (H-0094 decision 10, named by the rounds 13-14 monitor.)
        # The existing priority is kept exactly: the canonical spelling wins
        # when both are written. `_pop_by_identity` is deliberately **not** used
        # here -- it refuses two spellings with different values, and this
        # module has an accepted decision that `seed` takes priority over
        # `random_state` (`test_lgbm_defaults.py`). Changing a refusal as a side
        # effect of a naming tidy-up is not this commit's business.
        for canonical in ("seed", "verbosity"):
            supplied = {
                name: user_params.pop(name)
                for name in list(user_params)
                if name in accepted_spellings(canonical)
            }
            if supplied:
                user_params[canonical] = supplied.get(
                    canonical, next(iter(supplied.values()))
                )
        # H-0079: respect user/Optuna-supplied objective when task-compatible.
        # Pre-H-0079 this value was silently stripped, so default_space
        # tune trials sampling e.g. "fair" actually trained with the task
        # default. Reject cross-task injections explicitly with CONFIG_INVALID.
        user_objective, _ = _pop_by_identity(user_params, "objective")
        if user_objective is not None:
            _check_objective_compatible(self.task, user_objective)
            params["objective"] = user_objective
        # Allow user-specified metric; fall back to task default if absent/empty
        # Accepts str, list[str], or list[str | dict] (H-0065 MetricEntry).
        user_metric, _ = _pop_by_identity(user_params, "metric")
        feval_list: list[Any] = []
        feval_display_names: list[str] = []
        if user_metric:
            if isinstance(user_metric, (str, dict)):
                user_metric = [user_metric]
            # Filter out empty strings (dicts are always kept)
            user_metric = [m for m in user_metric if m]
            if user_metric:
                # Resolve: translate LizyML names, split native vs feval,
                # and validate against whitelist (H-0064, H-0065)
                native, feval_list, feval_display_names = resolve_metrics(
                    user_metric, self.task, num_class=self.num_class
                )
                params["metric"] = native if native else "None"
        # H-0094: LightGBM resolves aliases and prefers the canonical spelling
        # when both are present, and the defaults above are canonical. So a
        # user parameter written as an alias was merged in beside its own
        # default and then ignored -- measured: `model.params {"eta": 0.07}`
        # trained at 0.001, the default. Drop a default the user has named
        # under any spelling, so LightGBM never sees two spellings of one
        # parameter and the result does not depend on which it prefers.
        user_identities = {LGBM_CANONICAL_NAME.get(name, name) for name in user_params}
        for key in [
            key
            for key in params
            if key not in user_params
            and LGBM_CANONICAL_NAME.get(key, key) in user_identities
        ]:
            del params[key]
        params.update(user_params)

        # H-0079 L5: invariant guard — if a user objective was supplied and
        # task-compatible, it must survive _build_params(). Catches future
        # regressions to the silent-strip pattern even if someone refactors
        # the body. Disabled under `python -O` (production), active in
        # dev / test / CI.
        assert (  # noqa: S101 — defensive contract guard, see H-0079
            user_objective is None or params["objective"] == user_objective
        ), (
            f"H-0079 invariant violated: user-supplied objective="
            f"'{user_objective}' did not survive _build_params() "
            f"(got '{params.get('objective')}'). Likely regression to "
            f"the silent-strip pattern."
        )

        return params, num_boost_round, feval_list, feval_display_names

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
