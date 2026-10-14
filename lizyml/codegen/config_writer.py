"""config_writer — build config.json for codegen export."""

from __future__ import annotations

from typing import Any

from lizyml.codegen.values import plain_values

#: The libraries whose versions decide the split, the sort and the category
#: codes (H-0120 premises), by distribution name.
VERSIONED_LIBRARIES = ("lightgbm", "numpy", "pandas", "scikit-learn")


def library_versions() -> dict[str, str]:
    """The installed versions of :data:`VERSIONED_LIBRARIES`."""
    import lightgbm
    import numpy
    import pandas
    import sklearn

    return {
        "lightgbm": lightgbm.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
    }


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
    target_classes: list[Any] | None = None,
    split: dict[str, Any] | None = None,
    calibration_params: dict[str, Any] | None = None,
    unseen_policy: str = "mode",
    inner_valid: dict[str, Any] | None = None,
    sample_weight: str | None = None,
    declared_categories: dict[str, list[Any]] | None = None,
    categorical_rule: dict[str, Any] | None = None,
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
        unseen_policy: The encoder's ``unseen_policy`` from the fitted
            pipeline state (H-0104).
        inner_valid: The refit's inner-validation split, or ``None`` when the
            refit had no validation set (H-0120).
        sample_weight: ``"balanced"`` when the refit trained with per-row
            balanced weights, else ``None`` (H-0120).
        declared_categories: Columns that were ``category`` dtype at fit, with
            their declared categories (H-0120).
        categorical_rule: The data builder's cast rule, ``{"explicit": [...],
            "auto": bool}`` from ``features.categorical`` and
            ``features.auto_categorical``; defaults to LizyML's defaults
            (H-0120 amendment 3).

    Returns:
        Dict ready for ``json.dump()``.

    Raises:
        LizyMLError: With ``SERIALIZATION_FAILED`` when a target label or a
            declared category is outside the accepted value types
            (:mod:`lizyml.codegen.values`).
    """
    config_norm = run_meta.get("config_normalized", {})
    task = config_norm.get("task", "regression")
    data_cfg = config_norm.get("data", {})
    target_col = data_cfg.get("target", data_cfg.get("target_col", "y"))

    # H-0070: serialise target encoder so train.py can re-encode and
    # predict.py can decode int codes back to original labels.
    # H-0120: the labels must come back from JSON as the same values.
    target_encoder_block: dict[str, Any] = {
        "needs_encoding": bool(target_classes),
        "classes": plain_values(list(target_classes), where="target labels")
        if target_classes
        else [],
    }
    declared = {
        col: plain_values(list(cats), where=f"declared categories of column {col!r}")
        for col, cats in (declared_categories or {}).items()
    }

    return {
        # ── Meta (read-only, _ prefix) ──
        "_generated_by": f"lizyml {run_meta['lizyml_version']}",
        "_run_id": run_meta["run_id"],
        "_task": task,
        "_target_col": target_col,
        "_timestamp": run_meta["timestamp"],
        # H-0120: the generated train.py warns when a version differs.
        "_versions": library_versions(),
        # ── Features ──
        "feature_names": list(feature_names),
        "categorical_features": list(categorical_features),
        # H-0104: the unseen-category policy the fit applied. The generated
        # train.py writes it into the pipeline state it rebuilds, so a retrain
        # does not fall back to predict.py's "nan" default.
        "unseen_policy": unseen_policy,
        # H-0120: restored before the pipeline is fitted, so a CSV that lost
        # the dtype gets the declared codes back.
        "declared_categories": declared,
        # H-0120 amendment 3: which columns LizyML's data builder casts to
        # `category` before the encoder.
        "categorical_rule": {
            "explicit": list((categorical_rule or {}).get("explicit", [])),
            "auto": bool((categorical_rule or {}).get("auto", True)),
        },
        # ── LightGBM ──
        "lgbm_params": dict(lgbm_params),
        "num_boost_round": num_boost_round,
        # H-0120: the refit's validation split and patience, each written
        # always (null when absent) and read without a default; the callback
        # exists only when both are present.
        "inner_valid": dict(inner_valid) if inner_valid is not None else None,
        "early_stopping_rounds": early_stopping_rounds,
        "validation_ratio": validation_ratio,
        "sample_weight": sample_weight,
        "seed": seed,
        # ── Feval metrics (H-0066) ──
        "feval_metrics": list(feval_metrics) if feval_metrics else [],
        # ── Target encoder (H-0070) ──
        "target_encoder": target_encoder_block,
        # ── Calibration ──
        "calibration_method": calibration_method,
        "calibration_n_splits": calibration_n_splits,
        # H-0100: the prepared calibration.params, so a retrain rebuilds the
        # calibrator with the settings the fit used (H-0059).
        "calibration_params": dict(calibration_params or {}),
        # ── Split reproduction for retrain OOF (#228) ──
        "split": split,
    }
