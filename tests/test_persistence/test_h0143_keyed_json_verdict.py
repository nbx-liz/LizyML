"""H-0143 acceptance 2 and 3: the json verdict reads a keyed column as the
generated ``predict.py`` does.

- Models whose keyed column holds a ``np.timedelta64`` beside a ``np.uint64``
  are accepted by the three exits, and the loaded models and the generated
  ``predict.py`` predict as the original. Through ``Model.fit`` a column with
  a uint64 of 2**63 or more is a stand-in column (H-0133) and was accepted
  before too; ``4`` beside ``timedelta64(4, 'ps')`` was refused for some hash
  seeds.
- Negative controls: the runtime-encoder verdict refuses the JSON projection
  pandas cannot build -- ``OverflowError`` for 2**63 and above in a fixed
  category order, ``ValueError`` (uniqueness) for the ``4`` pair under
  ``PYTHONHASHSEED=0``.
- On a keyed column, a JSON projection that loses a category, reorders the
  categories or moves the mode is still refused with the same kind.
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.features.encoders.categorical_encoder import CategoricalEncoder
from lizyml.features.encoders.state_changes import category_state_changes
from lizyml.persistence import export as persistence_export
from lizyml.persistence._category_preflight import _encoder_through_json, state_changes
from tests._helpers import make_config
from tests.test_codegen._retrain_harness import load_module

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

PAIRS = {
    "uint64 above 2**63": [np.timedelta64(4, "s"), np.uint64(2**63 + 5)],
    "uint64 4 beside 4 ps": [np.timedelta64(4, "ps"), np.uint64(4)],
}


def _frame(values: list[Any], n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pool = [*values, "x"]
    # The first rows hold the values in this order, which the fitted
    # categories keep (first appearance): [timedelta64, uint64, "x"] is the
    # order whose JSON projection the runtime encoder refuses for the large
    # uint64 whatever the hash seed.
    drawn = [pool[i] for i in rng.integers(0, len(pool), n - len(pool))]
    column = pd.Series([*pool, *drawn], dtype=object)
    df = pd.DataFrame({"c": column, "n": rng.normal(size=n)})
    df["target"] = (column == "x").astype(float) + rng.normal(scale=0.1, size=n)
    return df


def _fitted(values: list[Any]) -> tuple[Model, pd.DataFrame]:
    cfg = make_config("regression", n_estimators=3, n_splits=2)
    cfg["features"] = {"categorical": ["c"]}
    model = Model(cfg)
    df = _frame(values)
    model.fit(data=df)
    return model, df


@pytest.mark.parametrize("label", sorted(PAIRS))
def test_the_exits_accept_a_keyed_column_the_generated_code_reads(
    tmp_path: Path, label: str
) -> None:
    model, df = _fitted(PAIRS[label])
    X = df.drop(columns=["target"])
    expected = np.asarray(model.predict(X).pred)

    # The three exits (H-0136), each before anything else is checked.
    state = model._get_fit_state()  # noqa: SLF001
    model.export(tmp_path / "a")
    persistence_export(
        tmp_path / "b",
        state.fit_result,
        model._refit_result,  # noqa: SLF001
        state.cfg.model_dump(),
        state.cfg.task,
        declared_categories=state.declared_categories,
    )
    model.export_code(tmp_path / "c")

    for saved in ("a", "b"):
        loaded = Model.load(tmp_path / saved)
        np.testing.assert_allclose(
            np.asarray(loaded.predict(X).pred), expected, rtol=1e-7
        )
    module = load_module(
        tmp_path / "c" / "predict.py", f"h0143_{label.replace(' ', '_')}"
    )
    np.testing.assert_allclose(
        np.asarray(module.predict(X)["pred"]), expected, rtol=1e-7
    )


def _ordered_state(categories: list[Any]) -> dict[str, Any]:
    """A keyed state of column ``c`` with *categories* in this order."""
    train = pd.DataFrame({"c": pd.Series([*categories, *categories], dtype=object)})
    state = CategoricalEncoder(unseen_policy="error").fit(train, ["c"]).get_state()
    assert state["identity_columns"] == ["c"]
    return {**state, "categories": {"c": categories}, "modes": {"c": categories[0]}}


def test_the_runtime_encoder_verdict_refused_the_large_uint64() -> None:
    """Negative control: in this order pandas' monotonic check overflows on
    the JSON-projected int, whatever the hash seed (measured on seeds 0-7)."""
    state = _ordered_state([np.timedelta64(4, "s"), np.uint64(2**63 + 5), "x"])
    projected = _encoder_through_json(state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runtime = category_state_changes(state, projected)
        generated = category_state_changes(state, projected, generated=True)
    assert [(c.kind, c.exception) for c in runtime] == [("raises", "OverflowError")]
    assert generated == []
    assert state_changes({"encoder": state}) == []


_PS_CONTROL = """
import warnings
import numpy as np
from lizyml.features.encoders.state_changes import category_state_changes
from lizyml.persistence._category_preflight import _encoder_through_json, state_changes
from tests.test_persistence.test_h0143_keyed_json_verdict import _ordered_state
warnings.simplefilter("ignore")
state = _ordered_state([np.timedelta64(4, "ps"), "x", np.uint64(4)])
projected = _encoder_through_json(state)
runtime = category_state_changes(state, projected)
print([(c.kind, c.exception) for c in runtime])
generated = category_state_changes(state, projected, generated=True)
print(generated, state_changes({"encoder": state}))
"""


def test_the_runtime_encoder_verdict_refused_uint64_4_beside_4_ps() -> None:
    """Negative control: pandas' uniqueness check takes 4 and timedelta64(4,
    'ps') for one category only for some hash seeds, so it runs with seed 0
    (measured to refuse; seeds 3, 6 and 7 do not)."""
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", _PS_CONTROL],
        cwd=root,
        env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == ["[('raises', 'ValueError')]", "[] []"]


def _keyed_state() -> dict[str, Any]:
    train = pd.DataFrame({"c": pd.Series(["a", "b", "b", "x", "x", "x"], dtype=object)})
    state = CategoricalEncoder(unseen_policy="error").fit(train, ["c"]).get_state()
    assert state["identity_columns"] == ["c"]
    return state


def _with(state: dict[str, Any], **changes: Any) -> dict[str, Any]:
    return {**state, **changes}


@pytest.mark.parametrize(
    ("kind", "change"),
    [
        (
            "unseen",
            lambda s: _with(
                s, categories={"c": [v for v in s["categories"]["c"] if v != "a"]}
            ),
        ),
        (
            "codes",
            lambda s: _with(s, categories={"c": list(reversed(s["categories"]["c"]))}),
        ),
        ("mode", lambda s: _with(s, modes={"c": "a"})),
    ],
)
def test_a_keyed_projection_that_answers_differently_is_still_refused(
    kind: str, change: Any
) -> None:
    state = _keyed_state()
    projected = change(state)
    runtime = category_state_changes(state, projected)
    generated = category_state_changes(state, projected, generated=True)
    assert [c.kind for c in generated] == [kind]
    assert [c.kind for c in runtime] == [kind]
