"""generator — orchestrate full codegen export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lizyml.calibration.base import BaseCalibratorAdapter
from lizyml.codegen.artifact_writer import write_artifacts
from lizyml.codegen.config_writer import build_config
from lizyml.codegen.templates import (
    render_predict_py,
    render_requirements_txt,
    render_test_equivalence_py,
    render_train_py,
)
from lizyml.estimators.lgbm.adapter import LGBMAdapter


def generate_code(
    *,
    output_dir: str | Path,
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
    model_adapter: LGBMAdapter,
    pipeline_state: dict[str, Any],
    calibrator: BaseCalibratorAdapter | None,
    feval_metrics: list[dict[str, Any]] | None = None,
    target_classes: list[Any] | None = None,
) -> Path:
    """Generate LizyML-independent training and prediction code.

    Creates the following structure::

        {output_dir}/
        ├── config.json
        ├── train.py
        ├── predict.py
        ├── test_equivalence.py
        ├── requirements.txt
        └── artifacts/
            ├── model.txt
            ├── pipeline_state.json
            ├── calibrator.json      (binary + calibrator only)
            └── calibrator_model.txt (isotonic only)

    Args:
        output_dir: Root directory for the exported code.
        run_meta: RunMeta-like dict with version, run_id, timestamp, config_normalized.
        feature_names: Ordered feature column names.
        categorical_features: Names of categorical features.
        lgbm_params: LightGBM parameters.
        num_boost_round: Number of boosting rounds.
        early_stopping_rounds: Early stopping patience.
        validation_ratio: Holdout validation fraction.
        seed: Random seed.
        calibration_method: Calibration method name or None.
        calibration_n_splits: CV splits for OOF calibration.
        model_adapter: Fitted LGBMAdapter.
        pipeline_state: Serializable pipeline state dict.
        calibrator: Fitted calibrator or None.
        feval_metrics: List of feval metric descriptors (H-0066).
            Each dict has ``name``, ``params``, ``greater_is_better``,
            ``needs_proba``.  Defaults to ``[]``.

    Returns:
        The resolved output directory path.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    # Build config
    config = build_config(
        run_meta=run_meta,
        feature_names=feature_names,
        categorical_features=categorical_features,
        lgbm_params=lgbm_params,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        validation_ratio=validation_ratio,
        seed=seed,
        calibration_method=calibration_method,
        calibration_n_splits=calibration_n_splits,
        feval_metrics=feval_metrics,
        target_classes=target_classes,
    )

    # Write artifacts (config.json, model.txt, pipeline_state.json, calibrator)
    write_artifacts(
        output_dir=root,
        config=config,
        model_adapter=model_adapter,
        pipeline_state=pipeline_state,
        calibrator=calibrator,
    )

    # Write source files
    (root / "train.py").write_text(render_train_py())
    (root / "predict.py").write_text(render_predict_py())
    (root / "test_equivalence.py").write_text(render_test_equivalence_py())
    (root / "requirements.txt").write_text(render_requirements_txt())

    return root
