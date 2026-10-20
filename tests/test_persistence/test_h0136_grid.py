"""H-0136 acceptance criterion 6: the rule's verdict matches what really happens.

Over the H-0133 sample population (every sample of ``SAMPLES`` alone and in
pairs, beside the string ``"x"``), the H-0136 verdict per serializer must
equal the outcome of the real serializer and consumer:

- joblib: the fitted encoder through ``joblib.dump`` / ``joblib.load``
  (``compress=3``, as the artifact) transforms the learned rows;
- json: ``convert_pipeline_state`` writes ``pipeline_state.json`` into a
  generated project and the generated ``predict.py``'s ``transform`` encodes
  the learned rows.

A serializer "breaks" when the transform raises or a code differs from the
fitted encoder's. The rows are P1's reference frame: the learned categories
in the state's order, as one batch. In another batch order pandas can map a
NumPy scalar equal to 1 onto a nested tuple's position in memory, before
any save (``docs/audits/2026-10-h0136/probes/pandas_tuple_lookup.py``); that
in-memory defect is outside H-0136.
"""

from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.codegen.artifact_writer import convert_pipeline_state
from lizyml.features.encoders.categorical_encoder import CategoricalEncoder
from lizyml.persistence._category_preflight import state_changes
from tests._helpers import make_config
from tests.test_codegen._retrain_harness import load_module
from tests.test_core.test_category_values import SAMPLES

pytestmark = pytest.mark.slow

_SAMPLES = [v for vs in SAMPLES.values() for v in vs]
_COLUMNS = [(v,) for v in _SAMPLES] + list(itertools.combinations(_SAMPLES, 2))


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, Path]:
    """A generated project whose ``pipeline_state.json`` each case rewrites."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "c": pd.Series(rng.choice(["a", "b"], 200), dtype=object),
            "target": rng.integers(0, 2, 200),
        }
    )
    cfg = make_config("binary", n_estimators=3, n_splits=2)
    cfg["features"] = {"categorical": ["c"], "unseen_policy": "error"}
    model = Model(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data=df)
    project = tmp_path_factory.mktemp("h0136_grid") / "code"
    model.export_code(project)
    return load_module(project / "predict.py", "h0136_grid_predict"), project


def _codes(frame: pd.DataFrame) -> list[float]:
    codes = frame["c"].cat.codes.to_numpy(dtype=float)
    codes[codes < 0] = np.nan
    return codes.tolist()


def _same(got: Any, want: list[float]) -> bool:
    return bool(np.array_equal(np.asarray(got, dtype=float), want, equal_nan=True))


class _Unseen:
    """A value no category equals: the row unseen_policy="mode" replaces."""


def _with_unseen(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"c": pd.Series([*df["c"].tolist(), _Unseen()], dtype=object)})


def _mode_code(encoder: CategoricalEncoder, df: pd.DataFrame) -> float | None:
    """What the mode does in memory: the unseen row's code, or ``None`` when
    the in-memory encoder itself fails on that batch (then it is not judged)."""
    encoder.unseen_policy = "mode"
    try:
        return _codes(encoder.transform(_with_unseen(df)))[-1]
    except Exception:  # noqa: BLE001 -- not judged, as in the rule
        return None
    finally:
        encoder.unseen_policy = "error"


def _joblib_breaks(
    encoder: CategoricalEncoder,
    df: pd.DataFrame,
    want: list[float],
    want_mode: float | None,
    tmp: Path,
) -> bool:
    path = tmp / "encoder.pkl"
    joblib.dump(encoder, path, compress=3)
    try:
        loaded = joblib.load(path)
        if not _same(_codes(loaded.transform(df)), want):
            return True
        if want_mode is None:
            return False
        loaded.unseen_policy = "mode"
        got = _codes(loaded.transform(_with_unseen(df)))[-1]
        return not _same([got], [want_mode])
    except Exception:  # noqa: BLE001 -- the outcome measured
        return True


def _json_breaks(
    module: Any,
    project: Path,
    state: dict[str, Any],
    df: pd.DataFrame,
    want: list[float],
    want_mode: float | None,
) -> bool:
    config = {"feature_names": ["c"], "categorical_features": ["c"]}
    path = project / "artifacts" / "pipeline_state.json"
    try:
        for policy in ("error", "mode"):
            if policy == "mode" and want_mode is None:
                return False
            codegen_state = convert_pipeline_state(
                {"feature_names": ["c"], "encoder": {**state, "unseen_policy": policy}},
                config,
            )
            path.write_text(
                json.dumps(codegen_state, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if policy == "error":
                if not _same(module.transform(df)["c"], want):
                    return True
            else:
                got = module.transform(_with_unseen(df))["c"].tolist()[-1]
                return not _same([got], [want_mode])
    except Exception:  # noqa: BLE001 -- the outcome measured
        return True
    return False


def test_the_verdict_matches_the_real_serializers(
    generated: tuple[Any, Path], tmp_path: Path
) -> None:
    module, project = generated
    mismatches: list[str] = []
    judged = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for column in _COLUMNS:
            train = pd.DataFrame(
                {"c": pd.Series([*column, *column, "x"], dtype=object)}
            )
            try:
                encoder = CategoricalEncoder(unseen_policy="error").fit(train, ["c"])
                state = encoder.get_state()
                # P1's reference frame: the learned categories, in the
                # state's order, as one batch (H-0136 実装の記録).
                df = pd.DataFrame(
                    {"c": pd.Series(list(state["categories"]["c"]), dtype=object)}
                )
                want = _codes(encoder.transform(df))
            except Exception:  # noqa: BLE001 -- the fit itself fails: not a case
                continue
            judged += 1
            verdict = {c.serializer for c in state_changes({"encoder": state})}
            want_mode = _mode_code(encoder, df)
            real = set()
            if _joblib_breaks(encoder, df, want, want_mode, tmp_path):
                real.add("joblib")
            if _json_breaks(module, project, state, df, want, want_mode):
                real.add("json")
            if verdict != real:
                mismatches.append(
                    f"{column!r}: verdict {sorted(verdict)} real {sorted(real)}"
                )
    assert judged > 5000
    assert not mismatches, "\n".join(mismatches[:20])
