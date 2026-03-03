"""Pydantic v2 schema definitions for LizyML configuration.

All models use extra="forbid" to catch typos as CONFIG_INVALID errors.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

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


SplitConfig = Annotated[
    KFoldConfig | StratifiedKFoldConfig | GroupKFoldConfig | TimeSeriesConfig,
    Field(discriminator="method"),
]


# ---------------------------------------------------------------------------
# ModelConfig (discriminated union)
# ---------------------------------------------------------------------------


class LGBMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["lgbm"]
    params: dict[str, Any] = {}


ModelConfig = Annotated[LGBMConfig, Field(discriminator="name")]


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


class HoldoutInnerValidConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["holdout"]
    ratio: float = 0.1
    random_state: int = 42


class EarlyStoppingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    rounds: int = 50
    inner_valid: HoldoutInnerValidConfig | None = None


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
