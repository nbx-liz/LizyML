"""artifact_writer — write config.json and artifacts/ for codegen export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lizyml.calibration.base import BaseCalibratorAdapter
from lizyml.estimators.lgbm.adapter import LGBMAdapter


def write_artifacts(
    *,
    output_dir: str | Path,
    config: dict[str, Any],
    model_adapter: LGBMAdapter,
    pipeline_state: dict[str, Any],
    calibrator: BaseCalibratorAdapter | None,
) -> Path:
    """Write config.json and artifacts/ directory for codegen export.

    Args:
        output_dir: Root directory for the exported code.
        config: Config dict from :func:`build_config`.
        model_adapter: Fitted LGBMAdapter to export.
        pipeline_state: Serializable pipeline state dict.
        calibrator: Fitted calibrator or None.

    Returns:
        The resolved output directory path.
    """
    root = Path(output_dir)
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    # config.json
    with open(root / "config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # model.txt
    model_adapter.save_model_text(artifacts / "model.txt")

    # pipeline_state.json
    with open(artifacts / "pipeline_state.json", "w") as f:
        json.dump(pipeline_state, f, indent=2, ensure_ascii=False)

    # calibrator
    if calibrator is not None:
        params = calibrator.export_params()
        with open(artifacts / "calibrator.json", "w") as f:
            json.dump(params, f, indent=2)

        # Isotonic: also save the Booster model file
        if hasattr(calibrator, "save_model_text"):
            calibrator.save_model_text(artifacts / "calibrator_model.txt")

    return root
