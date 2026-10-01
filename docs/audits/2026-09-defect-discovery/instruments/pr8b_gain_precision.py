"""PR 8b: is the gain-importance drift across export -> load LightGBM's text format?

``importance("gain")`` differs by about 1e-7 relative after ``Model.load()``
(see ``pr8b_load_census.py``). This isolates the cause. A ``lightgbm.Booster``
pickles through ``model_to_string``, and that text writes each tree's
``split_gain=`` with six significant digits, so the gain importance a loaded
booster sums is a sum of rounded gains. Predictions do not move, because leaf
values are written at full precision.

Steps, all on the fold-0 booster of a real ``Model.fit``:

1. the booster against ``Booster(model_str=booster.model_to_string())``;
2. the adapter against ``pickle.loads(pickle.dumps(adapter))``;
3. ``Model.predict`` and the OOF predictions before export and after load.

The tolerance a permanent check may use follows from the format, not from the
observation: six significant digits round each split's gain by at most half a
unit in the sixth digit, a relative error of at most 5e-6, and a feature's gain
is a sum of positive split gains, so its relative error is bounded by the same
5e-6. The observed maximum is printed to show it sits under that bound.

Run:

    uv run python docs/audits/2026-09-defect-discovery/instruments/pr8b_gain_precision.py
"""

from __future__ import annotations

import pickle
import sys
import tempfile
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from lizyml import Model  # noqa: E402
from tests._helpers import (  # noqa: E402
    make_binary_df,
    make_config,
    make_multiclass_df,
    make_regression_df,
)

warnings.simplefilter("ignore")

DATA = {
    "regression": make_regression_df,
    "binary": make_binary_df,
    "multiclass": make_multiclass_df,
}


def main() -> int:
    worst = 0.0
    for task, make in DATA.items():
        df = make(n=200)
        model = Model(
            make_config(task, n_estimators=10, n_splits=2, num_threads=1), data=df
        )
        model.fit()
        adapter = model.fit_result.models[0]
        booster = adapter.get_native_model()

        text = booster.model_to_string()
        gains = next(
            line for line in text.splitlines() if line.startswith("split_gain=")
        )
        from_text = lgb.Booster(model_str=text)
        before = booster.feature_importance("gain")
        after = from_text.feature_importance("gain")
        nonzero = before != 0
        rel = np.abs(after[nonzero] - before[nonzero]) / np.abs(before[nonzero])
        worst = max(worst, float(rel.max(initial=0.0)))

        pickled = pickle.loads(pickle.dumps(adapter))
        print(f"[{task}]")
        print(f"  first tree's {gains[:60]} ...")
        print(f"  booster gain        {before}")
        print(f"  text round trip     {after}")
        print(f"  max relative error  {float(rel.max(initial=0.0)):.3e}")
        pickled_gain = np.array(list(pickled.importance("gain").values()))
        print(
            "  pickled adapter's gain == text round trip's: "
            f"{bool(np.array_equal(pickled_gain, after))}"
        )

        X = df.drop(columns=["target"])
        with tempfile.TemporaryDirectory() as tmp:
            model.export(Path(tmp) / "a")
            loaded = Model.load(Path(tmp) / "a")
            same_pred = np.array_equal(model.predict(X).pred, loaded.predict(X).pred)
        same_oof = np.array_equal(
            model.fit_result.oof_pred, loaded.fit_result.oof_pred
        )
        print(f"  predict equal after load: {same_pred}; OOF equal: {same_oof}")

    print()
    print(f"largest relative gain error over all tasks: {worst:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
