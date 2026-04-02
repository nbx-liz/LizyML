"""config_writer — build config.json for codegen export."""

from __future__ import annotations

from typing import Any


def build_config(
    *,
    run_meta: dict[str, Any],
    feature_names: list[str],
    categorical_features: list[str],
    lgbm_params: dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int | None,
    validation_ratio: float,
    seed: int,
    calibration_method: str | None,
    calibration_n_splits: int,
    feval_metrics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build config.json content as an ordered dict.

    The returned dict is JSON-serializable and follows the key ordering:
    meta (``_`` prefix) → features → lgbm → feval → calibration.

    Args:
        run_meta: Dict with ``lizyml_version``, ``run_id``, ``timestamp``,
            and ``config_normalized`` (containing ``task`` and ``data.target_col``).
        feature_names: Ordered feature column names.
        categorical_features: Names of categorical features.
        lgbm_params: LightGBM parameters (excluding num_boost_round).
        num_boost_round: Number of boosting rounds.
        early_stopping_rounds: Early stopping patience (None to disable).
        validation_ratio: Fraction for holdout validation.
        seed: Random seed.
        calibration_method: Calibration method name or None.
        calibration_n_splits: Number of CV splits for OOF calibration.
        feval_metrics: List of feval metric descriptors (H-0066).  Each dict
            has keys ``name``, ``params``, ``greater_is_better``,
            ``needs_proba``.  Defaults to ``[]``.

    Returns:
        Dict ready for ``json.dump()``.
    """
    config_norm = run_meta.get("config_normalized", {})
    task = config_norm.get("task", "regression")
    data_cfg = config_norm.get("data", {})
    target_col = data_cfg.get("target", data_cfg.get("target_col", "y"))

    return {
        # ── Meta (read-only, _ prefix) ──
        "_generated_by": f"lizyml {run_meta['lizyml_version']}",
        "_run_id": run_meta["run_id"],
        "_task": task,
        "_target_col": target_col,
        "_timestamp": run_meta["timestamp"],
        # ── Features ──
        "feature_names": list(feature_names),
        "categorical_features": list(categorical_features),
        # ── LightGBM ──
        "lgbm_params": dict(lgbm_params),
        "num_boost_round": num_boost_round,
        "early_stopping_rounds": early_stopping_rounds,
        "validation_ratio": validation_ratio,
        "seed": seed,
        # ── Feval metrics (H-0066) ──
        "feval_metrics": list(feval_metrics) if feval_metrics else [],
        # ── Calibration ──
        "calibration_method": calibration_method,
        "calibration_n_splits": calibration_n_splits,
    }
