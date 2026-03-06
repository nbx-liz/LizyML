"""Pydantic v2 schema definitions for LizyML configuration.

All models use extra="forbid" to catch typos as CONFIG_INVALID errors.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

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
    random_state: int = 42
    shuffle: bool = True


class StratifiedKFoldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["stratified_kfold"]
    n_splits: int = 5
    random_state: int = 42


class GroupKFoldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["group_kfold"]
    n_splits: int = 5


class TimeSeriesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["time_series"]
    n_splits: int = 5
    gap: int = 0
    train_size_max: int | None = None
    test_size_max: int | None = None


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
                "use 'purge_gap' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            data["purge_gap"] = data.pop("purge_window")
        if "embargo_pct" in data and "embargo" not in data:
            warnings.warn(
                "purged_time_series key 'embargo_pct' is deprecated; "
                "use 'embargo' (int, obs count) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            data["embargo"] = int(data.pop("embargo_pct"))
        if "gap" in data and "embargo" not in data:
            warnings.warn(
                "purged_time_series key 'gap' is deprecated; use 'embargo' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            data["embargo"] = int(data.pop("gap"))
        return data


class GroupTimeSeriesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["group_time_series"]
    n_splits: int = 5
    gap: int = 0
    train_size_max: int | None = None
    test_size_max: int | None = None


SplitConfig = Annotated[
    KFoldConfig
    | StratifiedKFoldConfig
    | GroupKFoldConfig
    | TimeSeriesConfig
    | PurgedTimeSeriesConfig
    | GroupTimeSeriesConfig,
    Field(discriminator="method"),
]


# ---------------------------------------------------------------------------
# ModelConfig (discriminated union)
# ---------------------------------------------------------------------------


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
        if not (0 < self.num_leaves_ratio <= 1):
            raise ValueError(
                f"num_leaves_ratio must be in (0, 1], got {self.num_leaves_ratio}"
            )
        if self.min_data_in_leaf_ratio is not None and not (
            0 < self.min_data_in_leaf_ratio < 1
        ):
            raise ValueError(
                f"min_data_in_leaf_ratio must be in (0, 1), "
                f"got {self.min_data_in_leaf_ratio}"
            )
        if self.min_data_in_bin_ratio is not None and not (
            0 < self.min_data_in_bin_ratio < 1
        ):
            raise ValueError(
                f"min_data_in_bin_ratio must be in (0, 1), "
                f"got {self.min_data_in_bin_ratio}"
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


class EarlyStoppingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    rounds: int = 150
    inner_valid: InnerValidConfig | None = None
    validation_ratio: float | None = 0.1
    _inner_valid_explicit: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def _resolve_validation_ratio(self) -> EarlyStoppingConfig:
        iv_explicit = "inner_valid" in self.model_fields_set
        vr_explicit = "validation_ratio" in self.model_fields_set
        if iv_explicit and vr_explicit:
            # Allow round-trip: model_dump() produces both; consistent is OK
            if (
                isinstance(self.inner_valid, HoldoutInnerValidConfig)
                and self.inner_valid.ratio == self.validation_ratio
            ):
                return self
            raise ValueError(
                "Specify either 'validation_ratio' or 'inner_valid', not both."
            )
        if iv_explicit:
            self._inner_valid_explicit = True
            return self
        if self.validation_ratio is not None:
            self.inner_valid = HoldoutInnerValidConfig(
                method="holdout", ratio=self.validation_ratio
            )
        return self


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
    model_config = ConfigDict(extra="forbid")

    metrics: list[str] = []


# ---------------------------------------------------------------------------
# CalibrationConfig
# ---------------------------------------------------------------------------


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["platt", "isotonic", "beta"] = "platt"
    n_splits: int = 5
    params: dict[str, Any] = {}


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
