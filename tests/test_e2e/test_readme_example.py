"""README example test.

Executes the actual fenced code blocks from ``README.md`` so that drift between
the documented Quick Start and the real API is caught. Previously this test
hardcoded its own copy of the example, so README regressions went undetected
(H-0178 item 4).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from lizyml import Model

_README = Path(__file__).resolve().parents[2] / "README.md"
_PY_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _python_blocks() -> list[str]:
    return _PY_BLOCK.findall(_README.read_text(encoding="utf-8"))


def _quick_start_block() -> str:
    # The Quick Start block is the first python block that trains a Model.
    for block in _python_blocks():
        if "model.fit(" in block and "model.evaluate(" in block:
            return block
    raise AssertionError("Quick Start block not found in README.md")


def _codegen_block() -> str:
    for block in _python_blocks():
        if "export_code(" in block:
            return block
    raise AssertionError("Codegen block not found in README.md")


def test_readme_quick_start_block_executes(monkeypatch, tmp_path: Path) -> None:
    """The Quick Start snippet must run verbatim from README.md."""
    # export()/load() in the snippet use relative paths — run in a temp cwd.
    monkeypatch.chdir(tmp_path)
    namespace: dict = {}
    exec(compile(_quick_start_block(), str(_README), "exec"), namespace)

    # The snippet binds ``model`` and ``metrics``; assert the documented shape.
    assert "raw" in namespace["metrics"]
    # The same ``model`` powers the codegen one-liner shown later in the README.
    exec(compile(_codegen_block(), str(_README), "exec"), namespace)
    assert (tmp_path / "deploy" / "my_model" / "predict.py").exists()


def test_binary_readme_example() -> None:
    """Binary classification minimal smoke example (not sourced from README)."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, 200),
            "feat_b": rng.uniform(-1, 1, 200),
        }
    )
    df["target"] = (df["feat_a"] > 5).astype(int)

    model = Model(
        {
            "config_version": 1,
            "task": "binary",
            "data": {"target": "target"},
            "split": {"method": "kfold", "n_splits": 3, "random_state": 42},
            "model": {"name": "lgbm", "params": {"n_estimators": 20}},
            "training": {"seed": 0},
        }
    )
    model.fit(data=df)
    X_new = df.drop(columns=["target"]).iloc[:5]
    pred = model.predict(X_new)
    assert pred.proba is not None
    assert pred.pred.shape == (5,)
