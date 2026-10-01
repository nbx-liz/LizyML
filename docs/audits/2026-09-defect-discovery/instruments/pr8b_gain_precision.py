"""PR 8b: is the gain-importance drift across export -> load LightGBM's text format?

``importance("gain")`` differs by about 1e-7 to 1e-6 relative after
``Model.load()`` (see ``pr8b_load_census.py``). This isolates the cause. A
``lightgbm.Booster`` pickles through ``model_to_string``; that text writes each
tree's ``split_gain=`` with six significant digits, and the loader reads it back
into a binary32 ``float``. A loaded booster's gain importance is a sum of those
twice-rounded gains.

Steps:

1. For the fold-0 booster of a real ``Model.fit`` in three tasks: the booster
   against ``Booster(model_str=booster.model_to_string())``; the pickled adapter's
   gain against the text round trip's; ``Model.predict`` and the OOF predictions
   before export and after load (bit-identical in these fixtures -- an
   observation, not a guarantee this probe establishes); the smallest split gain
   in every tree (the bound needs them non-negative).
2. The bound, swept over positive finite binary32 values: random mantissas at
   every decimal exponent from 1e-45 to 1e38 (subnormals included), plus
   leading-digit-1 values next to every half-unit boundary of the sixth digit.
   Each is formatted with ``%g`` (six significant digits, as the text shows),
   read back as float32, and compared with the bound.

The bound, per split gain g: rounding to six significant digits gives d with
|d - g| <= 5e-6 |g|; reading d back into binary32 adds at most 2**-24 |d| while
the result is normal, and at most 2**-150 (half the subnormal spacing) when it
is subnormal, where a relative bound does not exist. So
|after - g| <= 5.0596e-6 |g| + 2**-150, with 5.0596e-6 = (1 + 5e-6)(1 + 2**-24) - 1.
A feature's gain is a sum of n non-negative split gains G, so it moves by at most
5.0596e-6 G + n 2**-150, and n 2**-150 < 2**-126 (binary32's smallest normal)
for any model with fewer than 2**24 splits.

The review history of this bound: the first version said 5e-6 and left out the
binary32 read (design review round 1 measured 5.054e-6); the second gave the
relative term alone and swept normal values only (round 2 measured 5.602e-6 on a
subnormal gain of about 1e-39). The absolute term is what the subnormal range
needs.

Run:

    uv run python docs/audits/2026-09-defect-discovery/instruments/pr8b_gain_precision.py
"""

from __future__ import annotations

import pickle
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

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
BOUND = (1 + 5e-6) * (1 + 2.0**-24) - 1
ABS_TERM = 2.0**-150


def _split_gains(node: dict[str, Any]) -> list[float]:
    if "split_gain" not in node:
        return []
    return [
        float(node["split_gain"]),
        *_split_gains(node["left_child"]),
        *_split_gains(node["right_child"]),
    ]


def models() -> float:
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
        pickled_gain = np.array(list(pickled.importance("gain").values()))
        all_gains = [
            gain
            for fold in model.fit_result.models
            for tree in fold.get_native_model().dump_model()["tree_info"]
            for gain in _split_gains(tree["tree_structure"])
        ]
        print(f"[{task}]")
        print(f"  first tree's {gains[:60]} ...")
        print(f"  booster gain        {before}")
        print(f"  text round trip     {after}")
        print(f"  max relative error  {float(rel.max(initial=0.0)):.3e}")
        print(
            "  pickled adapter's gain == text round trip's: "
            f"{bool(np.array_equal(pickled_gain, after))}"
        )
        print(
            f"  split gains over all folds: {len(all_gains)}, "
            f"smallest {min(all_gains):.6g}"
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
    return worst


def sweep() -> tuple[float, float, int]:
    """Return (worst relative error over normal values, worst ratio of error
    to the bound over all values, number of values swept)."""
    rng = np.random.default_rng(0)
    worst_rel = 0.0
    worst_ratio = 0.0
    swept = 0
    tiny = np.finfo(np.float32).tiny
    near_half = 1.0 + (np.arange(1000) + 0.5) * 1e-5
    for exponent in range(-45, 39):
        # Random mantissas, plus the worst case on purpose: a leading digit of
        # 1 (largest half-unit relative to the value) just off each half-unit
        # boundary of the sixth digit, where round 1's counterexample sat.
        mantissas = np.concatenate(
            [rng.uniform(1.0, 10.0, size=2000), near_half * (1 - 1e-7), near_half]
        )
        values = (mantissas * 10.0**exponent).astype(np.float32)
        values = values[np.isfinite(values) & (values > 0)]
        if not len(values):
            continue
        swept += len(values)
        wide = values.astype(np.float64)
        back = np.array([np.float32(float(f"{float(v):g}")) for v in values])
        err = np.abs(back.astype(np.float64) - wide)
        normal = values >= tiny
        if normal.any():
            worst_rel = max(worst_rel, float((err[normal] / wide[normal]).max()))
        worst_ratio = max(worst_ratio, float((err / (BOUND * wide + ABS_TERM)).max()))
    return worst_rel, worst_ratio, swept


def main() -> int:
    worst_models = models()
    worst_rel, worst_ratio, swept = sweep()
    print()
    print(f"relative term (1 + 5e-6) * (1 + 2**-24) - 1 = {BOUND:.6e}; absolute term 2**-150")
    print(f"largest relative gain error, real models: {worst_models:.3e}")
    print(f"sweep: {swept} positive finite float32 values, 1e-45..1e38, subnormals included")
    print(f"  largest relative error over normal values: {worst_rel:.6e}")
    print(f"  largest error / (relative term * value + absolute term): {worst_ratio:.6f}")
    print(f"  every value within the bound: {worst_ratio <= 1.0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
