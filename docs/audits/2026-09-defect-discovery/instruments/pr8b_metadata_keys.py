"""PR 8b: the top-level ``metadata.json`` key set an export writes, per lifecycle.

Run at ``develop`` before H-0109 this is the legacy key set. H-0109 adds one
key, and ``test_the_record_is_the_only_new_key`` checks the implemented exporter
adds nothing else, so an artifact with that key deleted has the key set a
pre-H-0109 export writes. That is a statement about keys, not bytes: the values
under the other keys come from whichever exporter wrote them. The pickles do not
change: H-0109 touches neither ``FitResult`` nor ``RefitResult``.

Run:

    uv run python docs/audits/2026-09-defect-discovery/instruments/pr8b_metadata_keys.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from lizyml import Model  # noqa: E402
from tests._helpers import make_binary_df, make_config  # noqa: E402

warnings.simplefilter("ignore")

SPACE = {
    "validation_ratio": {"type": "categorical", "choices": [0.45], "category": "training"}
}


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("fit", "tune_fit", "fit_tune"):
            cfg = make_config(
                "binary", n_estimators=10, n_splits=2, num_threads=1, tuning_n_trials=1
            )
            cfg["tuning"]["optuna"]["space"] = dict(SPACE)
            model = Model(cfg, data=make_binary_df(n=200))
            steps = {"fit": ["fit"], "tune_fit": ["tune", "fit"], "fit_tune": ["fit", "tune"]}
            for step in steps[name]:
                getattr(model, step)()
            out = Path(tmp) / name
            model.export(out)
            meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
            tuning = meta.get("tuning")
            print(f"{name:<9} keys={sorted(meta)}")
            print(
                f"{'':<9} tuning.best_training_params="
                f"{None if tuning is None else tuning['best_training_params']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
