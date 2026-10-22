"""H-0138 acceptance 2 (encoder cases) and 11: today's answers, frozen.

Every case of the oracle for the running pandas version is rebuilt from its
id and run through the steps the oracle used (``astype("category")`` as
``dataframe_builder`` does, then ``CategoricalEncoder.fit`` and two transform
batches). The categories with their types, the mode, ``stand_in_columns`` and
the codes must equal the frozen file; a temporal column must give exactly the
frozen outcome, a failure included, unless H-0139's closed overlay lists it
(then its ``after`` outcome; the temporal answers run the real builder
route). Nothing here is recomputed from the code under test: the answers are
computed once, in a subprocess under the seed
the oracle was made with (``PYTHONHASHSEED=0``), because today's order of a
tuple beside a frozenset follows string hashing.

H-0144: the oracle and its overlay are the running pandas version's, or, in
the latest-deps CI job only, the nearest older version of the same minor
(``_h0138_oracle.resolve``). There, the model cases are skipped when numpy or
lightgbm differ from the oracle's, decided before any model is computed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pytest

from tests.test_core.test_category_values import SAMPLES
from tests.test_features import _h0138_oracle as h

ROOT = Path(__file__).resolve().parents[2]
RESOLVED = h.resolve()
#: ``None`` on a pandas without a selectable oracle: the case lists are then
#: empty and ``test_an_oracle_exists_for_this_pandas`` fails.
ORACLE = h.oracle_or_none(RESOLVED)
CASES = ORACLE["cases"] if ORACLE is not None else []
TEMPORAL = ORACLE["temporal_cases"] if ORACLE is not None else []
#: Whether the model cases run on this lane (H-0144 対応方針 2).
MODELS = ORACLE is not None and h.compare_models(ORACLE["meta"], RESOLVED.mode)


def test_the_oracle_mode_is_allowed_here() -> None:
    assert RESOLVED.error is None, RESOLVED.error


def test_an_oracle_exists_for_this_pandas() -> None:
    assert ORACLE is not None, (
        f"no H-0138 oracle for pandas {RESOLVED.running} (mode {RESOLVED.mode}): "
        f"{h.oracle_path(RESOLVED.running)}"
    )


@pytest.fixture(scope="module")
def answers() -> dict[str, Any]:
    env = {
        **os.environ,
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": str(ROOT),
        h.MODELS_ENV: "1" if MODELS else "0",
    }
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
    assert meta["pandas"] == RESOLVED.version
    available = h.oracle_versions()
    assert h.select_version(RESOLVED.running, available, RESOLVED.mode) == (
        RESOLVED.version
    )
    assert RESOLVED.exact or RESOLVED.mode == h.SAME_MINOR
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


def _skip_models() -> None:
    assert ORACLE is not None, "no H-0138 oracle selected for this lane"
    if not MODELS:
        meta = ORACLE["meta"]
        pytest.skip(
            f"latest-deps lane: the oracle {RESOLVED.version} was made with "
            f"numpy/lightgbm {meta['numpy']}/{meta['lightgbm']}, this lane runs "
            f"{np.__version__}/{lightgbm.__version__}"
        )


def test_the_lane_runs_the_oracles_numpy_and_lightgbm() -> None:
    _skip_models()
    h.model_versions_match(ORACLE["meta"])


def test_models_are_computed_only_when_compared(request: pytest.FixtureRequest) -> None:
    """H-0144 acceptance 5: a skipped model case is never computed."""
    if ORACLE is None:
        pytest.skip("no oracle: test_an_oracle_exists_for_this_pandas fails")
    answers = request.getfixturevalue("answers")
    assert answers["models_computed"] == (len(MODEL_CASES) if MODELS else 0)


@pytest.mark.parametrize("case", MODEL_CASES, ids=[c["id"] for c in MODEL_CASES])
def test_a_faithful_column_trains_the_same_model_as_today(
    case: dict[str, Any], answers: dict[str, Any]
) -> None:
    _skip_models()
    h.model_versions_match(ORACLE["meta"])
    assert answers["models"][case["id"]] == case["model"]


#: H-0139's closed list of temporal cases it changes on purpose. ``None`` on
#: a pandas without one: ``test_an_overlay_exists_for_this_pandas`` fails.
OVERLAY = h.overlay_or_none(RESOLVED)
CHANGED = {c["id"]: c for c in OVERLAY["changes"]} if OVERLAY is not None else {}

#: H-0139 対応方針 7: the S set named per pandas version (2.0.0 decided by
#: the generator from the same two rules; HISTORY.md ``## H-0139``).
S_NAMED = {
    "3.0.1": ["p22-23", "p22-36", "p23-35", "p24-38", "p60-103"],
    "2.3.3": [
        "p22-23",
        "p22-36",
        "p22-40",
        "p23-35",
        "p23-41",
        "p24-38",
        "p36-41",
        "p62-105",
    ],
    "2.0.0": [
        "p22-23",
        "p22-36",
        "p22-40",
        "p23-35",
        "p23-41",
        "p24-38",
        "p36-41",
        "p62-105",
    ],
}


def _case_indices(case_id: str) -> set[int]:
    return {int(i) for i in case_id[1:].split("-")} if case_id[0] in "sp" else set()


def test_an_overlay_exists_for_this_pandas() -> None:
    assert OVERLAY is not None, (
        f"no H-0139 overlay for pandas {RESOLVED.version}: "
        f"{h.overlay_path(RESOLVED.version)}"
    )
    assert OVERLAY["meta"]["pandas"] == RESOLVED.version


def test_the_overlay_is_the_closed_union_of_its_rules() -> None:
    """Membership follows from the frozen oracle, never from the library."""
    assert OVERLAY is not None and ORACLE is not None
    changes = OVERLAY["changes"]
    ids = [c["id"] for c in changes]
    assert len(ids) == len(set(ids))
    frozen = {c["id"]: c["outcome"] for c in TEMPORAL}
    assert all(c["before"] == frozen[c["id"]] for c in changes)
    samples = [v for vs in SAMPLES.values() for v in vs]
    calendar = {
        i
        for i, v in enumerate(samples)
        if type(v) is np.timedelta64 and np.datetime_data(v.dtype)[0] in ("Y", "M")
    }
    y = {cid for cid in frozen if _case_indices(cid) & calendar}
    r = {cid for cid, out in frozen.items() if out.get("fails_at") == "cast"} - y
    # H-0144 rule T: the cast held the column, the old lookup failed.
    t = {cid for cid, out in frozen.items() if out.get("fails_at") == "transform"} - y
    assert {c["rule"] for c in changes} <= {"Y", "R", "S", "T"}
    by_rule = {
        rule: {c["id"] for c in changes if c["rule"] == rule}
        for rule in ("Y", "R", "S", "T")
    }
    assert RESOLVED.version is not None
    named = S_NAMED[RESOLVED.version]
    assert by_rule["Y"] == y
    assert by_rule["R"] == r
    assert by_rule["T"] == t
    assert sorted(by_rule["S"]) == named
    # Nothing outside the union may override a frozen outcome.
    assert set(ids) == y | r | t | set(named)
    # An empty rule is left out of the counts (the overlays before H-0144).
    counts = {k: len(v) for k, v in by_rule.items() if v}
    assert OVERLAY["meta"]["counts"] == counts


@pytest.mark.parametrize("case", TEMPORAL, ids=[c["id"] for c in TEMPORAL])
def test_a_temporal_column_behaves_as_frozen_or_as_h0139_decided(
    case: dict[str, Any], answers: dict[str, Any]
) -> None:
    expected = (
        CHANGED[case["id"]]["after"] if case["id"] in CHANGED else case["outcome"]
    )
    assert answers["temporal_cases"][case["id"]] == expected


h.SUMMARY.update(
    {
        "pandas": RESOLVED.running,
        "numpy": np.__version__,
        "lightgbm": lightgbm.__version__,
        "pyarrow": h.pyarrow_version(),
        "mode": RESOLVED.mode or "strict",
        "oracle": RESOLVED.version,
        "exact": RESOLVED.exact,
        "encoder_cases": len(CASES),
        "temporal_cases": len(TEMPORAL),
        "model_cases": len(MODEL_CASES),
        "models": "compared" if MODELS else ("skipped" if ORACLE else "none"),
    }
)
