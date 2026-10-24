"""H-0145 acceptance 3a (#391): the generated train.py refuses, before writing
pipeline_state.json, a category value whose frozenset does not read back.

A project exported from an ordinary model is retrained on rows holding the
value. The negative control swaps the checked writer for the unchecked one and
shows the state would be written and could not be loaded again.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests.test_codegen.test_h0138_identity_copy import _modules, _project
from tests.test_core.test_h0145_round_trip import CASES, REASON, collapsing

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _retrain_rows(df: pd.DataFrame, value: Any) -> pd.DataFrame:
    raw = df.drop(columns=["target"]).copy()
    raw["c"] = pd.Series(
        [value if i % 2 == 0 else "x" for i in range(len(raw))],
        dtype=object,
        index=raw.index,
    )
    return raw


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("name", list(CASES))
def test_the_generated_retrain_refuses_before_writing(
    name: str, tmp_path: Path
) -> None:
    value = collapsing(name)
    _, df, project = _project(["a", "b"], tmp_path)
    train, predict = _modules(project)
    state_file = project / "artifacts" / "pipeline_state.json"
    before = _digest(state_file)
    raw = train.cast_categoricals(_retrain_rows(df, value)[train.CFG["feature_names"]])
    with pytest.raises(ValueError, match=REASON):
        train.fit_pipeline(raw)
    assert _digest(state_file) == before
    predict._load_pipeline()  # the project still loads


@pytest.mark.parametrize("name", list(CASES))
def test_without_the_check_the_state_is_written_and_cannot_be_read(
    name: str, tmp_path: Path
) -> None:
    """Negative control: the unchecked writer is what the check guards."""
    value = collapsing(name)
    _, df, project = _project(["a", "b"], tmp_path)
    train, predict = _modules(project)
    train._encode_value = train._encode_raw
    state_file = project / "artifacts" / "pipeline_state.json"
    before = _digest(state_file)
    raw = train.cast_categoricals(_retrain_rows(df, value)[train.CFG["feature_names"]])
    train.fit_pipeline(raw)
    assert _digest(state_file) != before
    with pytest.raises(ValueError):
        predict._load_pipeline()
