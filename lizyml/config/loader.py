"""Config loader for LizyML.

Supports loading from dict, JSON file, and YAML file.
Applies alias normalization and environment variable overrides before validation.
Also provides Config → Spec conversion.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from lizyml.core.exceptions import ErrorCode, LizyMLError

from .schema import LizyMLConfig

# ---------------------------------------------------------------------------
# Alias normalization
# ---------------------------------------------------------------------------

_SPLIT_METHOD_ALIASES: dict[str, str] = {
    "k-fold": "kfold",
    "kfold": "kfold",
    "stratified-kfold": "stratified_kfold",
    "stratifiedkfold": "stratified_kfold",
    "group-kfold": "group_kfold",
    "groupkfold": "group_kfold",
    "time-series": "time_series",
    "timeseries": "time_series",
    "purged-time-series": "purged_time_series",
    "purgedtimeseries": "purged_time_series",
    "group-time-series": "group_time_series",
    "grouptimeseries": "group_time_series",
}

_ENV_PREFIX = "LIZYML__"
_ENV_SEP = "__"


def _normalize_split_default(raw: dict[str, Any]) -> dict[str, Any]:
    """Inject default split config if absent.

    For classification tasks (binary/multiclass), defaults to stratified_kfold.
    For regression, defaults to kfold.
    """
    if "split" not in raw:
        task = raw.get("task", "regression")
        if task in ("binary", "multiclass"):
            raw = {
                **raw,
                "split": {
                    "method": "stratified_kfold",
                    "n_splits": 5,
                    "random_state": 42,
                },
            }
        else:
            raw = {
                **raw,
                "split": {
                    "method": "kfold",
                    "n_splits": 5,
                    "random_state": 42,
                    "shuffle": True,
                },
            }
    return raw


def _normalize_split_method(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize split.method aliases (e.g. 'k-fold' → 'kfold')."""
    split = raw.get("split")
    if isinstance(split, dict):
        method = split.get("method")
        if isinstance(method, str):
            normalized = _SPLIT_METHOD_ALIASES.get(method.lower())
            if normalized is not None and normalized != method:
                raw = {**raw, "split": {**split, "method": normalized}}
    return raw


_KNOWN_MODEL_NAMES = ("lgbm",)


def _merge_env_stray_key(model: dict[str, Any]) -> dict[str, Any]:
    """Merge a stray nested key left by env override into the model dict.

    When env overrides add ``model.lgbm.params.x=1`` after the model dict
    already has ``"name": "lgbm"``, we get a stray nested key.
    This merges the nested dict back into the top level.
    """
    name = model["name"]
    if name not in model or not isinstance(model[name], dict):
        return model
    inner: dict[str, Any] = model[name]
    merged = {k: v for k, v in model.items() if k != name}
    # Deep merge params
    if "params" in inner:
        existing_params: dict[str, Any] = merged.get("params", {})
        merged["params"] = {**existing_params, **inner["params"]}
    for k, v in inner.items():
        if k != "params":
            merged[k] = v
    return merged


def _convert_blueprint_model(model: dict[str, Any]) -> dict[str, Any] | None:
    """Convert BLUEPRINT format ``{"lgbm": {...}}`` to ``{"name": "lgbm", ...}``."""
    for model_name in _KNOWN_MODEL_NAMES:
        if model_name in model:
            inner_cfg = model[model_name]
            if isinstance(inner_cfg, dict):
                return {"name": model_name, **inner_cfg}
    return None


def _normalize_model_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize model config to discriminated union form."""
    model = raw.get("model")
    if not isinstance(model, dict):
        return raw

    if "name" in model:
        return {**raw, "model": _merge_env_stray_key(model)}

    converted = _convert_blueprint_model(model)
    if converted is not None:
        return {**raw, "model": converted}
    return raw


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides with LIZYML__ prefix.

    Example: LIZYML__model__lgbm__params__learning_rate=0.01
    sets raw["model"]["lgbm"]["params"]["learning_rate"] = 0.01
    """
    import copy

    result = copy.deepcopy(raw)
    for key, value in os.environ.items():
        if not key.startswith(_ENV_PREFIX):
            continue
        path_str = key[len(_ENV_PREFIX) :]
        parts = [p.lower() for p in path_str.split(_ENV_SEP) if p]
        if not parts:
            continue
        node: Any = result
        for part in parts[:-1]:
            if not isinstance(node, dict):
                break
            if part not in node:
                node[part] = {}
            node = node[part]
        else:
            if isinstance(node, dict):
                last = parts[-1]
                # Attempt to coerce numeric/bool values
                node[last] = _coerce_env_value(value)
    return result


def _coerce_env_value(value: str) -> Any:
    """Coerce environment variable string to appropriate Python type."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


# ---------------------------------------------------------------------------
# Version gate
# ---------------------------------------------------------------------------

SUPPORTED_CONFIG_VERSIONS: list[int] = [1]


def _check_config_version(raw: dict[str, Any]) -> None:
    """Raise CONFIG_VERSION_UNSUPPORTED if config_version is unsupported."""
    version = raw.get("config_version")
    if isinstance(version, int) and version not in SUPPORTED_CONFIG_VERSIONS:
        raise LizyMLError(
            ErrorCode.CONFIG_VERSION_UNSUPPORTED,
            user_message=(
                f"config_version={version} is not supported. "
                f"Supported versions: {SUPPORTED_CONFIG_VERSIONS}"
            ),
            context={"config_version": version, "supported": SUPPORTED_CONFIG_VERSIONS},
        )


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_config(source: dict[str, Any] | str | Path) -> LizyMLConfig:
    """Load and validate a LizyML configuration.

    Args:
        source: A raw dict, a path to a JSON file, or a path to a YAML file.

    Returns:
        Validated ``LizyMLConfig`` instance.

    Raises:
        LizyMLError: With ``CONFIG_INVALID`` when validation fails.
        LizyMLError: With ``CONFIG_INVALID`` when the file cannot be read or parsed.
    """
    raw = _read_raw(source)
    _check_config_version(raw)
    raw = _normalize_split_default(raw)
    raw = _normalize_split_method(raw)
    raw = _normalize_model_config(raw)  # First pass: before env overrides
    raw = _apply_env_overrides(raw)
    raw = _normalize_model_config(raw)  # Second pass: env may add model.lgbm.* keys
    return _validate(raw)


def _read_raw(source: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(source, dict):
        return source

    path = Path(source)
    suffix = path.suffix.lower()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LizyMLError(
            ErrorCode.CONFIG_INVALID,
            user_message=f"Cannot read config file: {path}",
            debug_message=str(exc),
            cause=exc,
            context={"config_path": str(path)},
        ) from exc

    try:
        if suffix == ".json":
            return json.loads(text)  # type: ignore[no-any-return]
        if suffix in (".yaml", ".yml"):
            data = yaml.safe_load(text)
            if not isinstance(data, dict):
                raise LizyMLError(
                    ErrorCode.CONFIG_INVALID,
                    user_message=(
                        f"YAML file must contain a mapping at the top level: {path}"
                    ),
                    context={"config_path": str(path)},
                )
            return data
        raise LizyMLError(
            ErrorCode.CONFIG_INVALID,
            user_message=f"Unsupported config file format '{suffix}': {path}",
            context={"config_path": str(path)},
        )
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise LizyMLError(
            ErrorCode.CONFIG_INVALID,
            user_message=f"Failed to parse config file: {path}",
            debug_message=str(exc),
            cause=exc,
            context={"config_path": str(path)},
        ) from exc


def _validate(raw: dict[str, Any]) -> LizyMLConfig:
    try:
        return LizyMLConfig.model_validate(raw)
    except ValidationError as exc:
        raise LizyMLError(
            ErrorCode.CONFIG_INVALID,
            user_message=(
                "Config validation failed. Check for unknown keys or invalid values."
            ),
            debug_message=str(exc),
            cause=exc,
            context={"validation_errors": exc.errors()},
        ) from exc
