"""H-0134 size probe: how big a persisted tuning history is, and how long it takes to read.

One real tune (default search space, binary, small data) gives the actual trial
shape and the artifact's existing file sizes. Synthetic histories of N trials
are then built from that shape, written in the candidate form (ordered
``[name, encode_scalar(value)]`` params; trial number, score, state and round)
and read back. Sizes are bytes of compact JSON; times are the median of 5 runs.

Run: uv run python docs/audits/2026-10-h0134/probes/size_probe.py
"""

from __future__ import annotations

import json
import random
import statistics
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lizyml import Model
from lizyml.tuning.lineage_codec import decode_scalar, encode_dims, encode_scalar
from tests._helpers import make_config

warnings.simplefilter("ignore")


def encode_trial(t: Any) -> dict[str, Any]:
    return {
        "number": t.number,
        "params": [[k, encode_scalar(v)] for k, v in t.params.items()],
        "score": encode_scalar(t.score),
        "state": t.state,
        "round": t.round,
    }


def decode_trial(p: dict[str, Any]) -> tuple[Any, ...]:
    return (
        p["number"],
        {k: decode_scalar(v) for k, v in p["params"]},
        decode_scalar(p["score"]),
        p["state"],
        p["round"],
    )


def median_seconds(fn: Any) -> float:
    runs = []
    for _ in range(5):
        start = time.perf_counter()
        fn()
        runs.append(time.perf_counter() - start)
    return statistics.median(runs)


rng = np.random.default_rng(0)
n = 400
df = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(6)})
df["target"] = (df["x0"] + rng.normal(size=n) > 0).astype(int)
cfg = make_config("binary", n_estimators=20, n_splits=2, num_threads=1, tuning_n_trials=20)
model = Model(cfg, data=df)
model.tune()
model.fit()
result = model._tuning_result
assert result is not None
real = result.trials
print(f"real tune: {len(real)} trials, {len(real[0].params)} params per trial, "
      f"{len(result.rounds)} round summaries")
print("param types:", sorted({type(v).__name__ for t in real for v in t.params.values()}))

with tempfile.TemporaryDirectory() as tmp:
    out = Path(model.export(Path(tmp) / "a"))
    for f in sorted(out.iterdir()):
        print(f"artifact file {f.name}: {f.stat().st_size} bytes")

rounds_bytes = len(json.dumps(
    [encode_dims(r.space_snapshot) for r in result.rounds], separators=(",", ":")
))
print(f"space_snapshot of all rounds, encoded: {rounds_bytes} bytes")

random.seed(0)
print("synthetic history (trials, bytes, bytes/trial, write s, read+decode s):")
for count in (50, 500, 5000, 50000):
    trials = [encode_trial(random.choice(real)) for _ in range(count)]
    for i, t in enumerate(trials):
        t["number"] = i
    text = json.dumps({"trials": trials}, separators=(",", ":"), allow_nan=False)
    write = median_seconds(
        lambda: json.dumps({"trials": trials}, separators=(",", ":"), allow_nan=False)
    )
    read = median_seconds(lambda: [decode_trial(p) for p in json.loads(text)["trials"]])
    print(f"  {count:>6} {len(text):>11} {len(text) // count:>5} {write:.4f} {read:.4f}")
