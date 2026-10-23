"""#391 acceptance 3a: the generated train.py refuses before writing.

Exports a project from an ordinary keyed model, then retrains its
``fit_pipeline`` on rows holding a value whose frozenset does not read back
as written (NumPy 1.24: ``frozenset({np.uint64(4), np.timedelta64(4, "s")})``;
NumPy 2.x: ``frozenset({np.uint64(9265762820973641783), np.timedelta64(4,
"ms")})``). Reports whether ``fit_pipeline`` raised, whether
``pipeline_state.json`` changed, and whether the generated predict.py can
still load it. Also runs the generated codec copy over the round-2 values.

    PYTHONPATH=<tree> python -B docs/audits/2026-10-i391/probes/generated_writer.py <scratch>
"""

from __future__ import annotations

import hashlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import lizyml
from lizyml.core.category_values import refusal
from tests.test_codegen.test_h0133_codec_copy import COPY
from tests.test_codegen.test_h0138_identity_copy import _modules, _project

warnings.simplefilter("ignore")
print("lizyml", lizyml.__file__, "numpy", np.__version__, "pandas", pd.__version__)
CASES = {
    "pair": frozenset({np.uint64(4), np.timedelta64(4, "s")}),
    "big": frozenset({np.uint64(9265762820973641783), np.timedelta64(4, "ms")}),
}
for name, v in CASES.items():
    try:
        COPY._encode_value(v)
        gen = "admits"
    except ValueError as exc:
        gen = f"refuses ({exc})"
    print(f"{name}: library refusal={refusal(v)!r}; generated _encode_value {gen}; _in_v={COPY._in_v(v) if hasattr(COPY, '_in_v') else 'n/a'}")

scratch = Path(sys.argv[1])
scratch.mkdir(parents=True, exist_ok=True)
model, df, project = _project(["a", "b"], scratch)
train, predict = _modules(project)
state_file = project / "artifacts" / "pipeline_state.json"
for name, v in CASES.items():
    if refusal(v) is None:
        print(f"{name}: reads back on this NumPy; not a 3a case here")
        continue
    before = hashlib.sha256(state_file.read_bytes()).hexdigest()
    raw = df.drop(columns=["target"]).copy()
    raw["c"] = pd.Series([v if i % 2 == 0 else "x" for i in range(len(raw))], dtype=object)
    try:
        train.fit_pipeline(train.cast_categoricals(raw[train.CFG["feature_names"]]))
        outcome = "fit_pipeline returned"
    except ValueError as exc:
        outcome = f"fit_pipeline raised ValueError: {exc}"
    after = hashlib.sha256(state_file.read_bytes()).hexdigest()
    try:
        predict._load_pipeline()
        load = "generated _load_pipeline reads it"
    except Exception as exc:  # noqa: BLE001 -- the probe reports it
        load = f"generated _load_pipeline raises {type(exc).__name__}"
    print(f"{name}: {outcome}; pipeline_state.json {'unchanged' if before == after else 'REWRITTEN'}; {load}")
