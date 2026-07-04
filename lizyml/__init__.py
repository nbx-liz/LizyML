"""LizyML: config-driven ML analysis library."""

from lizyml._version import __version__, __version_tuple__
from lizyml.config.loader import load_config
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.model import Model
from lizyml.core.types.fit_result import FitResult
from lizyml.core.types.predict_result import PredictionResult
from lizyml.core.types.task import TaskType
from lizyml.core.types.tuning_result import (
    BoundaryDimStatus,
    BoundaryReport,
    RoundSummary,
    TuneProgressCallback,
    TuneProgressInfo,
    TuningResult,
)

__all__ = [
    "BoundaryDimStatus",
    "BoundaryReport",
    "ErrorCode",
    "FitResult",
    "LizyMLError",
    "Model",
    "PredictionResult",
    "RoundSummary",
    "TaskType",
    "TuneProgressCallback",
    "TuneProgressInfo",
    "TuningResult",
    "__version__",
    "__version_tuple__",
    "load_config",
]
