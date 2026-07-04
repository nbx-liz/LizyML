"""artifact_writer — write config.json and artifacts/ for codegen export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lizyml.calibration.base import BaseCalibratorAdapter
from lizyml.estimators.base import BaseEstimatorAdapter


def _convert_pipeline_state(
    state: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Convert LizyML pipeline state to codegen-compatible format.

    LizyML stores ``encoder.categories`` (list of known categories per column).
    Codegen ``predict.py`` expects ``category_mappings`` (str→int dicts).

    Also exports the encoder's ``unseen_policy`` and, per column, the integer
    code of the training mode (``unseen_codes``) so the generated ``predict.py``
    can reproduce the runtime ``unseen_policy="mode"`` behavior (#205). Without
    these, ``predict.py`` mapped unseen categories to NaN while the runtime
    ``CategoricalEncoder`` replaced them with the most frequent training
    category — a silent prediction divergence.
    """
    feature_names = state.get("feature_names", config.get("feature_names", []))
    categorical_features = config.get("categorical_features", [])

    # Build integer mappings from encoder categories
    encoder = state.get("encoder", {})
    categories = encoder.get("categories", {})
    modes = encoder.get("modes", {})
    unseen_policy = encoder.get("unseen_policy", "mode")
    mappings: dict[str, dict[str, int]] = {}
    unseen_codes: dict[str, int] = {}
    for col, cats in categories.items():
        mapping = {str(v): i for i, v in enumerate(cats)}
        mappings[col] = mapping
        mode_val = modes.get(col)
        # The mode is always one of the known categories, so its str form is a
        # key in ``mapping`` — record its code as the unseen replacement.
        if mode_val is not None and str(mode_val) in mapping:
            unseen_codes[col] = mapping[str(mode_val)]

    return {
        "feature_names": feature_names,
        "categorical_features": categorical_features,
        "category_mappings": mappings,
        "unseen_policy": unseen_policy,
        "unseen_codes": unseen_codes,
    }


def write_artifacts(
    *,
    output_dir: str | Path,
    config: dict[str, Any],
    model_adapter: BaseEstimatorAdapter,
    pipeline_state: dict[str, Any],
    calibrator: BaseCalibratorAdapter | None,
) -> Path:
    """Write config.json and artifacts/ directory for codegen export.

    Args:
        output_dir: Root directory for the exported code.
        config: Config dict from :func:`build_config`.
        model_adapter: Fitted estimator adapter to export. Currently only
            ``LGBMAdapter`` is supported by the codegen templates, but the
            type is widened to ``BaseEstimatorAdapter`` so callers in
            ``_model_persistence.py`` can stay estimator-agnostic (H-0073).
        pipeline_state: Serializable pipeline state dict.
        calibrator: Fitted calibrator or None.

    Returns:
        The resolved output directory path.
    """
    root = Path(output_dir)
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    # config.json (encoding="utf-8": ensure_ascii=False may emit non-ASCII,
    # which the Windows default cp1252 codec cannot encode — #180)
    with open(root / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # model.txt
    model_adapter.save_model_text(artifacts / "model.txt")

    # pipeline_state.json — convert LizyML format to codegen format
    codegen_state = _convert_pipeline_state(pipeline_state, config)
    with open(artifacts / "pipeline_state.json", "w", encoding="utf-8") as f:
        json.dump(codegen_state, f, indent=2, ensure_ascii=False)

    # calibrator
    if calibrator is not None:
        params = calibrator.export_params()
        with open(artifacts / "calibrator.json", "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)

        # Isotonic: also save the Booster model file
        if hasattr(calibrator, "save_model_text"):
            calibrator.save_model_text(artifacts / "calibrator_model.txt")

    return root
