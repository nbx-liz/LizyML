"""Pydantic v2 schema definitions for LizyML configuration.

All models use extra="forbid" to catch typos as CONFIG_INVALID errors.
"""

from __future__ import annotations

import warnings
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    computed_field,
    model_validator,
)

# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str | None = None
    target: str
    time_col: str | None = None
    group_col: str | None = None


# ---------------------------------------------------------------------------
# FeaturesConfig
# ---------------------------------------------------------------------------


class FeaturesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exclude: list[str] = []
    auto_categorical: bool = True
    categorical: list[str] = []


# ---------------------------------------------------------------------------
# SplitConfig (discriminated union)
# ---------------------------------------------------------------------------


class KFoldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["kfold"]
    n_splits: int = 5
    # None -> inherit training.seed at splitter-build time (H-0080).
    random_state: int | None = None
    shuffle: bool = True


class StratifiedKFoldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["stratified_kfold"]
    n_splits: int = 5
    # None -> inherit training.seed at splitter-build time (H-0080).
    random_state: int | None = None


class GroupKFoldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["group_kfold"]
    n_splits: int = 5


class StratifiedGroupKFoldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["stratified_group_kfold"]
    n_splits: int = 5
    # None -> inherit training.seed at splitter-build time (H-0080).
    random_state: int | None = None
    shuffle: bool = True


class TimeSeriesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["time_series"]
    n_splits: int = 5
    gap: int = 0
    train_size_max: int | None = None
    test_size_max: int | None = None


def _legacy_obs_count(key: str, value: Any) -> int:
    """Coerce a legacy purge/embargo value to an integer observation count.

    Rejects fractional values instead of silently truncating them to ``0`` — a
    leak-prevention parameter must never collapse to ``0`` (#210). Integer-valued
    inputs (``3`` or ``3.0``) are accepted; a fractional value (``0.05``) raises.
    """
    if isinstance(value, bool):
        raise ValueError(
            f"purged_time_series '{key}' must be an integer observation count, "
            f"got {value!r}."
        )
    if isinstance(value, int):
        return value
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"purged_time_series '{key}' must be an integer observation count, "
            f"got {value!r}."
        ) from None
    if as_float != int(as_float):
        raise ValueError(
            f"purged_time_series legacy '{key}'={value!r} is fractional; "
            "'embargo' is an integer observation count, not a fraction. It must "
            "not silently truncate to 0. Compute the row count explicitly "
            "(e.g. int(round(fraction * n_rows)))."
        )
    return int(as_float)


class PurgedTimeSeriesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["purged_time_series"]
    n_splits: int = 5
    purge_gap: int = 0
    embargo: int = 0
    train_size_max: int | None = None
    test_size_max: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_keys(cls, data: Any) -> Any:
        """Accept legacy keys with deprecation warning."""
        if not isinstance(data, dict):
            return data
        import warnings

        if "purge_window" in data and "purge_gap" not in data:
            warnings.warn(
                "purged_time_series key 'purge_window' is deprecated; "
                "use 'purge_gap' instead. Will be removed in v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            data["purge_gap"] = data.pop("purge_window")
        if "embargo_pct" in data and "embargo" not in data:
            warnings.warn(
                "purged_time_series key 'embargo_pct' is deprecated; "
                "use 'embargo' (int, obs count) instead. "
                "Will be removed in v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            data["embargo"] = _legacy_obs_count("embargo_pct", data.pop("embargo_pct"))
        if "gap" in data and "embargo" not in data:
            warnings.warn(
                "purged_time_series key 'gap' is deprecated; "
                "use 'embargo' instead. Will be removed in v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            data["embargo"] = _legacy_obs_count("gap", data.pop("gap"))
        return data


class GroupTimeSeriesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["group_time_series"]
    n_splits: int = 5
    gap: int = 0
    train_size_max: int | None = None
    test_size_max: int | None = None


# ---------------------------------------------------------------------------
# BlockedGroupKFold (H-0060): 2-axis CV (period × group)
# ---------------------------------------------------------------------------


class BlocksConfig(BaseModel):
    """Period axis configuration for blocked_group_kfold."""

    model_config = ConfigDict(extra="forbid")

    col: str
    cutoffs: list[Any] = Field(min_length=1)
    mode: Literal["expanding", "sliding"] = "expanding"
    train_window: int | None = None


class GroupCVConfig(BaseModel):
    """Group axis configuration for blocked_group_kfold."""

    model_config = ConfigDict(extra="forbid")

    col: str
    n_splits: int = 3
    stratify: Literal["auto"] | bool = "auto"
    shuffle: bool = True


class BlockedGroupKFoldConfig(BaseModel):
    """2-axis cross-validation: period blocks × group KFold (H-0060)."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["blocked_group_kfold"]
    blocks: BlocksConfig
    groups: GroupCVConfig
    min_train_rows: int = 10
    min_valid_rows: int = 5

    @model_validator(mode="after")
    def _validate_axes(self) -> BlockedGroupKFoldConfig:
        if self.blocks.col == self.groups.col:
            raise ValueError(
                f"blocks.col and groups.col must differ, both are '{self.blocks.col}'"
            )
        if self.blocks.mode == "sliding" and self.blocks.train_window is None:
            raise ValueError("train_window is required when mode is 'sliding'")
        if self.blocks.mode == "expanding" and self.blocks.train_window is not None:
            warnings.warn(
                "train_window is ignored when mode is 'expanding'",
                UserWarning,
                stacklevel=2,
            )
        return self


SplitConfig = Annotated[
    KFoldConfig
    | StratifiedKFoldConfig
    | GroupKFoldConfig
    | StratifiedGroupKFoldConfig
    | TimeSeriesConfig
    | PurgedTimeSeriesConfig
    | GroupTimeSeriesConfig
    | BlockedGroupKFoldConfig,
    Field(discriminator="method"),
]


# ---------------------------------------------------------------------------
# ModelConfig (discriminated union)
# ---------------------------------------------------------------------------


def _check_ratio(value: float | None, name: str, *, inclusive_upper: bool) -> None:
    """Validate a ratio parameter is in (0, 1] or (0, 1)."""
    if value is None:
        return
    hi_ok = value <= 1.0 if inclusive_upper else value < 1.0
    if not (value > 0 and hi_ok):
        bound = "1]" if inclusive_upper else "1)"
        raise ValueError(f"{name} must be in (0, {bound}, got {value}")


class LGBMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["lgbm"]
    params: dict[str, Any] = {}

    # Smart parameters (resolved at fit time)
    auto_num_leaves: bool = True
    num_leaves_ratio: float = 1.0
    min_data_in_leaf_ratio: float | None = 0.01
    min_data_in_bin_ratio: float | None = 0.01
    feature_weights: dict[str, float] | None = None
    balanced: bool | None = None

    @model_validator(mode="after")
    def _validate_smart_params(self) -> LGBMConfig:
        if self.auto_num_leaves and "num_leaves" in self.params:
            raise ValueError(
                "Cannot specify 'params.num_leaves' when 'auto_num_leaves' is True. "
                "Set 'auto_num_leaves: false' or remove 'num_leaves' from params."
            )
        if (
            self.min_data_in_leaf_ratio is not None
            and "min_data_in_leaf" in self.params
        ):
            raise ValueError(
                "Cannot specify both 'min_data_in_leaf_ratio' and "
                "'params.min_data_in_leaf'. Use one or the other."
            )
        if self.min_data_in_bin_ratio is not None and "min_data_in_bin" in self.params:
            raise ValueError(
                "Cannot specify both 'min_data_in_bin_ratio' and "
                "'params.min_data_in_bin'. Use one or the other."
            )
        _check_ratio(self.num_leaves_ratio, "num_leaves_ratio", inclusive_upper=True)
        _check_ratio(
            self.min_data_in_leaf_ratio, "min_data_in_leaf_ratio", inclusive_upper=False
        )
        _check_ratio(
            self.min_data_in_bin_ratio, "min_data_in_bin_ratio", inclusive_upper=False
        )
        if self.feature_weights:
            for k, v in self.feature_weights.items():
                if v <= 0:
                    raise ValueError(
                        f"feature_weights values must be > 0, got {v} for '{k}'"
                    )
        return self


ModelConfig = Annotated[LGBMConfig, Field(discriminator="name")]


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


class HoldoutInnerValidConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["holdout"]
    ratio: float = 0.1
    stratify: bool = False
    random_state: int = 42


class GroupHoldoutInnerValidConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["group_holdout"]
    ratio: float = 0.1
    random_state: int = 42


class TimeHoldoutInnerValidConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["time_holdout"]
    ratio: float = 0.1


InnerValidConfig = Annotated[
    HoldoutInnerValidConfig
    | GroupHoldoutInnerValidConfig
    | TimeHoldoutInnerValidConfig,
    Field(discriminator="method"),
]


_DEFAULT_VALIDATION_RATIO = 0.1


class EarlyStoppingConfig(BaseModel):
    """Early-stopping configuration.

    ``inner_valid`` is the canonical source of truth for the holdout
    fraction.  ``validation_ratio`` is exposed as a read-only computed
    field that mirrors ``inner_valid.ratio`` (H-0069).

    Legacy YAML inputs that supply only ``validation_ratio`` are
    transparently migrated into a ``HoldoutInnerValidConfig`` so the
    user-facing surface stays compatible.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    rounds: int = 150
    inner_valid: InnerValidConfig | None = None
    _inner_valid_explicit: bool = PrivateAttr(default=False)

    @model_validator(mode="wrap")
    @classmethod
    def _migrate_and_track_explicit(
        cls, data: Any, handler: Any
    ) -> EarlyStoppingConfig:
        """Translate legacy ``validation_ratio`` input + track explicitness.

        Behavior (H-0069):

        - Pure legacy input ``{"validation_ratio": 0.1}`` → migrates to
          ``inner_valid: {method: "holdout", ratio: 0.1}``;
          ``_inner_valid_explicit`` stays ``False`` so the factory's
          auto-resolve path keeps choosing the split-aware variant.
        - Explicit ``{"inner_valid": {...}}`` → kept as-is;
          ``_inner_valid_explicit`` becomes ``True``.
        - Round-trip dump ``{"inner_valid": {...}, "validation_ratio": x}``
          → ``validation_ratio`` is silently dropped (it is a computed
          field re-emitted by ``model_dump()``); explicitness mirrors
          the round-trip-without-vr semantics (False — auto-resolve).
        - Inconsistent values (``validation_ratio != inner_valid.ratio``)
          → ``ValueError`` to surface real conflicts.
        """
        user_explicit_inner_valid = False
        explicit_marker: bool | None = None
        if isinstance(data, dict):
            # A serialized explicitness marker (emitted by a prior
            # ``model_dump()``) is the round-trip source of truth: pop it before
            # ``handler`` since ``extra="forbid"`` — the same treatment the
            # computed ``validation_ratio`` gets. This keeps an explicit
            # ``inner_valid`` explicit across dump/reload instead of letting the
            # re-emitted ``validation_ratio`` flip it to auto-resolve (H-0086,
            # #203).
            if "inner_valid_explicit" in data:
                explicit_marker = bool(data.pop("inner_valid_explicit"))
            iv_in = data.get("inner_valid") is not None
            vr_present = "validation_ratio" in data
            user_explicit_inner_valid = iv_in and not vr_present
            if vr_present:
                legacy_ratio = data.pop("validation_ratio")
                iv_value = data.get("inner_valid")
                if iv_value is None:
                    if legacy_ratio is not None:
                        warnings.warn(
                            "`validation_ratio` is deprecated; use "
                            "`inner_valid.ratio` instead. "
                            "Will be removed in v1.0.",
                            DeprecationWarning,
                            stacklevel=2,
                        )
                        data["inner_valid"] = {
                            "method": "holdout",
                            "ratio": legacy_ratio,
                        }
                else:
                    iv_ratio = (
                        iv_value.get("ratio")
                        if isinstance(iv_value, dict)
                        else getattr(iv_value, "ratio", None)
                    )
                    if (
                        legacy_ratio is not None
                        and iv_ratio is not None
                        and legacy_ratio != iv_ratio
                    ):
                        raise ValueError(
                            "Specify either 'validation_ratio' or 'inner_valid', "
                            "not both."
                        )
        instance: EarlyStoppingConfig = handler(data)
        if instance.inner_valid is None:
            instance.inner_valid = HoldoutInnerValidConfig(
                method="holdout", ratio=_DEFAULT_VALIDATION_RATIO
            )
        instance._inner_valid_explicit = (
            explicit_marker
            if explicit_marker is not None
            else user_explicit_inner_valid
        )
        return instance

    @computed_field  # type: ignore[prop-decorator]
    @property
    def validation_ratio(self) -> float:
        """Holdout fraction; derived from ``inner_valid.ratio`` (H-0069)."""
        if self.inner_valid is None:
            return _DEFAULT_VALIDATION_RATIO
        return self.inner_valid.ratio

    @computed_field  # type: ignore[prop-decorator]
    @property
    def inner_valid_explicit(self) -> bool:
        """Round-trip marker for whether ``inner_valid`` was user-explicit.

        Emitted by ``model_dump()`` and honored on re-validation (popped like
        ``validation_ratio``) so a dumped/exported config preserves the user's
        explicit ``inner_valid`` choice instead of silently falling back to
        split-derived auto-resolution (H-0086, #203). Not a settable input
        field — it only mirrors the internal ``_inner_valid_explicit`` state.
        """
        return self._inner_valid_explicit


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()


# ---------------------------------------------------------------------------
# TuningConfig
# ---------------------------------------------------------------------------


class OptunaParamsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_trials: int = 50
    direction: Literal["minimize", "maximize"] = "minimize"
    timeout: float | None = None


class OptunaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    params: OptunaParamsConfig = OptunaParamsConfig()
    space: dict[str, Any] = {}


class TuningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    optuna: OptunaConfig = OptunaConfig()


# ---------------------------------------------------------------------------
# EvaluationConfig
# ---------------------------------------------------------------------------


class EvaluationConfig(BaseModel):
    """Evaluation configuration.

    ``metrics`` accepts both plain strings and parameterised dicts (H-0065)::

        metrics: ["auc", {"precision_at_k": {"k": 20}}]
    """

    model_config = ConfigDict(extra="forbid")

    metrics: list[str | dict[str, dict[str, Any]]] = []


# ---------------------------------------------------------------------------
# CalibrationConfig
# ---------------------------------------------------------------------------


_CALIBRATION_N_SPLITS_DEFAULT = 5


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["platt", "isotonic", "beta"] = "platt"
    n_splits: int = _CALIBRATION_N_SPLITS_DEFAULT
    params: dict[str, Any] = {}

    @model_validator(mode="before")
    @classmethod
    def _warn_deprecated_n_splits(cls, data: Any) -> Any:
        """Emit UserWarning when n_splits is explicitly set to a
        non-default value (H-0058).

        Only fires for dict inputs where ``n_splits`` differs from the
        default (5).  This avoids spurious warnings when
        ``model_dump()`` round-trips (e.g. ``Model.load()``) include
        the default value.
        """
        if (
            isinstance(data, dict)
            and "n_splits" in data
            and data["n_splits"] != _CALIBRATION_N_SPLITS_DEFAULT
        ):
            warnings.warn(
                "calibration.n_splits is deprecated and will be ignored. "
                "Calibration cross-fit now reuses outer CV splits (H-0058). "
                "Will be removed in v1.0.",
                UserWarning,
                stacklevel=2,
            )
        return data


# ---------------------------------------------------------------------------
# Top-level LizyMLConfig
# ---------------------------------------------------------------------------


class LizyMLConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_version: int
    task: Literal["regression", "binary", "multiclass"]
    data: DataConfig
    features: FeaturesConfig = FeaturesConfig()
    split: SplitConfig
    model: ModelConfig
    training: TrainingConfig = TrainingConfig()
    tuning: TuningConfig | None = None
    evaluation: EvaluationConfig = EvaluationConfig()
    calibration: CalibrationConfig | None = None
    output_dir: str | None = None
