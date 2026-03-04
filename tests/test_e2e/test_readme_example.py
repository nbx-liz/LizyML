"""README example test.

Verifies that the code shown in README.md runs without exceptions.
The example must work immediately after installation without extra config.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lizyml import Model


def test_readme_example_runs() -> None:
    """Minimal Model.fit → evaluate → predict example from README."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, 200),
            "feat_b": rng.uniform(-1, 1, 200),
            "target": rng.uniform(0, 1, 200),
        }
    )

    config = {
        "config_version": 1,
        "task": "regression",
        "data": {"target": "target"},
        "split": {"method": "kfold", "n_splits": 3, "random_state": 42},
        "model": {"name": "lgbm", "params": {"n_estimators": 20}},
        "training": {"seed": 0},
    }

    model = Model(config)
    result = model.fit(data=df)
    metrics = model.evaluate()
    X_new = df.drop(columns=["target"]).iloc[:10].reset_index(drop=True)
    predictions = model.predict(X_new)

    assert result is not None
    assert "raw" in metrics
    assert predictions.pred.shape == (10,)


def test_binary_readme_example() -> None:
    """Binary classification minimal example."""
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
