"""H-0138 acceptance 2 (encoder cases) and 11: today's answers, frozen.

Every case of the oracle for the running pandas version is rebuilt from its
id and run through the steps the oracle used (``astype("category")`` as
``dataframe_builder`` does, then ``CategoricalEncoder.fit`` and two transform
batches). The categories with their types, the mode, ``stand_in_columns`` and
the codes must equal the frozen file; a temporal column must give exactly the
frozen outcome, a failure included. Nothing here is recomputed from the code
under test: the answers are computed once, in a subprocess under the seed
the oracle was made with (``PYTHONHASHSEED=0``), because today's order of a
tuple beside a frozenset follows string hashing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests.test_features import _h0138_oracle as h

ROOT = Path(__file__).resolve().parents[2]
#: ``None`` on a pandas without an oracle (the latest-deps lane): the case
#: lists are then empty and ``test_an_oracle_exists_for_this_pandas`` fails.
ORACLE = h.oracle_or_none()
CASES = ORACLE["cases"] if ORACLE is not None else []
TEMPORAL = ORACLE["temporal_cases"] if ORACLE is not None else []


def test_an_oracle_exists_for_this_pandas() -> None:
    assert ORACLE is not None, (
        f"no H-0138 oracle for pandas {pd.__version__}: {h.oracle_path()}"
    )


@pytest.fixture(scope="module")
def answers() -> dict[str, Any]:
    env = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-m", "tests.test_features._h0138_answers"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=1200,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    out: dict[str, Any] = json.loads(result.stdout)
    return out


def test_the_oracle_matches_its_meta() -> None:
    meta = ORACLE["meta"]
    assert meta["pandas"] == pd.__version__
    assert len(CASES) == meta["encoder_cases"]
    assert len(TEMPORAL) == meta["temporal_cases"]
    known = set(h.all_probe_ids())
    assert {c["id"] for c in CASES if not c["id"].startswith("o-")} <= known
    assert {c["id"] for c in TEMPORAL} <= known


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_a_faithful_column_answers_as_today(
    case: dict[str, Any], answers: dict[str, Any]
) -> None:
    assert answers["cases"][case["id"]] == case["encoder"]


MODEL_CASES = [c for c in CASES if "model" in c]


def test_the_lane_runs_the_oracles_numpy_and_lightgbm() -> None:
    h.model_versions_match(ORACLE["meta"])


@pytest.mark.parametrize("case", MODEL_CASES, ids=[c["id"] for c in MODEL_CASES])
def test_a_faithful_column_trains_the_same_model_as_today(
    case: dict[str, Any], answers: dict[str, Any]
) -> None:
    h.model_versions_match(ORACLE["meta"])
    assert answers["models"][case["id"]] == case["model"]


@pytest.mark.parametrize("case", TEMPORAL, ids=[c["id"] for c in TEMPORAL])
def test_a_temporal_column_behaves_exactly_as_today(
    case: dict[str, Any], answers: dict[str, Any]
) -> None:
    assert answers["temporal_cases"][case["id"]] == case["outcome"]
