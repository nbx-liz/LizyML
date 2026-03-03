"""Unified exception hierarchy for LizyML."""

from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """Standardized error codes for all LizyML exceptions."""

    CONFIG_INVALID = "CONFIG_INVALID"
    CONFIG_VERSION_UNSUPPORTED = "CONFIG_VERSION_UNSUPPORTED"
    DATA_SCHEMA_INVALID = "DATA_SCHEMA_INVALID"
    DATA_FINGERPRINT_MISMATCH = "DATA_FINGERPRINT_MISMATCH"
    LEAKAGE_SUSPECTED = "LEAKAGE_SUSPECTED"
    LEAKAGE_CONFIRMED = "LEAKAGE_CONFIRMED"
    OPTIONAL_DEP_MISSING = "OPTIONAL_DEP_MISSING"
    MODEL_NOT_FIT = "MODEL_NOT_FIT"
    INCOMPATIBLE_COLUMNS = "INCOMPATIBLE_COLUMNS"
    UNSUPPORTED_TASK = "UNSUPPORTED_TASK"
    UNSUPPORTED_METRIC = "UNSUPPORTED_METRIC"
    METRIC_REQUIRES_PROBA = "METRIC_REQUIRES_PROBA"
    TUNING_FAILED = "TUNING_FAILED"
    CALIBRATION_NOT_SUPPORTED = "CALIBRATION_NOT_SUPPORTED"
    SERIALIZATION_FAILED = "SERIALIZATION_FAILED"
    DESERIALIZATION_FAILED = "DESERIALIZATION_FAILED"


class LizyMLError(Exception):
    """Base exception for all LizyML errors.

    Attributes:
        code: Standardized error code for programmatic handling.
        user_message: Human-readable message for end users.
        debug_message: Optional technical detail for developers.
        cause: Optional underlying exception that triggered this error.
        context: Optional dict with structured debugging info
                 (e.g. fold index, config path, column name).
    """

    def __init__(
        self,
        code: ErrorCode,
        user_message: str,
        *,
        debug_message: str | None = None,
        cause: BaseException | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.user_message = user_message
        self.debug_message = debug_message
        self.cause = cause
        self.context = context or {}
        super().__init__(user_message)

    def __str__(self) -> str:
        return f"[{self.code.value}] {self.user_message}"

    def __repr__(self) -> str:
        parts = [f"LizyMLError(code={self.code!r}, user_message={self.user_message!r}"]
        if self.debug_message:
            parts.append(f", debug_message={self.debug_message!r}")
        if self.context:
            parts.append(f", context={self.context!r}")
        if self.cause:
            parts.append(f", cause={self.cause!r}")
        parts.append(")")
        return "".join(parts)
