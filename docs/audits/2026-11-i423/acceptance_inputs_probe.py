"""H-0150 implementation prep: run each named acceptance input through today's tune admission.

For every space or fixed parameter that H-0150's acceptance criteria name, this
runs ``Model.tune`` once (2 trials, LightGBM training stubbed to a 1-tree,
4-leaf booster) and records whether admission refused it before any trial
(``CONFIG_INVALID`` and the like), or which trial states the study ended in.

Run from the repository root:
``.venv/bin/python <this file>``
"""

from __future__ import annotations

import copy
import sys
import warnings
from pathlib import Path
from typing import Any

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

import lightgbm  # noqa: E402

from lizyml import Model  # noqa: E402
from lizyml.core.exceptions import LizyMLError  # noqa: E402
from lizyml.estimators.lgbm.param_names import LGBM_CANONICAL_NAME  # noqa: E402
from tests._helpers import make_config, make_regression_df  # noqa: E402

_REAL = lightgbm.train
CALLS: list[dict[str, Any]] = []


def _stub(params: dict[str, Any], train_set: Any, num_boost_round: int, **kw: Any) -> Any:
    CALLS.append(dict(params))
    small = {
        k: v
        for k, v in params.items()
        if LGBM_CANONICAL_NAME.get(k, k)
        not in ("num_leaves", "max_depth", "device_type", "histogram_pool_size")
    }
    small.update(num_leaves=4, max_depth=2)
    return _REAL(small, train_set, num_boost_round=1, **kw)


lightgbm.train = _stub

BASE_SPACE = {"learning_rate": {"type": "float", "low": 0.05, "high": 0.1}}

CASES: list[tuple[str, dict[str, Any], dict[str, Any], bool]] = [
    # (name, model params, extra space dims, auto_num_leaves)
    ("fixed n_estimators=-1", {"n_estimators": -1}, {}, True),
    (
        "early_stopping_rounds [None, 40]",
        {},
        {"early_stopping_rounds": {"type": "categorical", "choices": [None, 40], "category": "training"}},
        True,
    ),
    (
        "validation_ratio [0, 0.3] + fixed device_type=gpu",
        {"device_type": "gpu"},
        {"validation_ratio": {"type": "float", "low": 0.0, "high": 0.3, "category": "training"}},
        True,
    ),
    (
        "validation_ratio [0, 0.3] (cpu)",
        {},
        {"validation_ratio": {"type": "float", "low": 0.0, "high": 0.3, "category": "training"}},
        True,
    ),
    ("int device dimension [1, 2]", {}, {"device": {"type": "int", "low": 1, "high": 2}}, True),
    (
        "device categorical [cpu, gpu]",
        {},
        {"device": {"type": "categorical", "choices": ["cpu", "gpu"]}},
        True,
    ),
    ("fixed device=cuda (alias)", {"device": "cuda"}, {}, True),
    (
        "histogram_pool_size dimension",
        {},
        {"histogram_pool_size": {"type": "float", "low": 64.0, "high": 256.0}},
        True,
    ),
    (
        "stepped num_leaves [16, 64] step 16, auto_num_leaves false",
        {},
        {"num_leaves": {"type": "int", "low": 16, "high": 64, "step": 16}},
        False,
    ),
    (
        "max_bin_by_feature dimension",
        {},
        {"max_bin_by_feature": {"type": "categorical", "choices": [[16], [32]]}},
        True,
    ),
    (
        "max_depth [3, 1024]",
        {},
        {"max_depth": {"type": "int", "low": 3, "high": 1024}},
        True,
    ),
]


def run(name: str, params: dict[str, Any], dims: dict[str, Any], auto: bool) -> str:
    n_est = params.pop("n_estimators", 3)
    cfg = make_config("regression", n_estimators=n_est, n_splits=2, **params)
    cfg["model"]["auto_num_leaves"] = auto
    space = copy.deepcopy(BASE_SPACE)
    space.update(dims)
    cfg["tuning"] = {
        "optuna": {"params": {"n_trials": 2}, "space": space, "space_mode": "replace"}
    }
    CALLS.clear()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = Model(cfg, data=make_regression_df(n=80)).tune()
    except LizyMLError as exc:
        reason = exc.context.get("reason") if isinstance(exc.context, dict) else None
        return (
            f"refused {exc.code.value} reason={reason} trains={len(CALLS)}: "
            f"{exc.user_message[:160]}"
        )
    except Exception as exc:  # noqa: BLE001 -- a census records every outcome
        return f"raised {type(exc).__name__}: {str(exc)[:160]} trains={len(CALLS)}"
    states = [t.state for t in result.trials]
    return f"ran states={states} trains={len(CALLS)}"


def main() -> int:
    for name, params, dims, auto in CASES:
        print(f"- {name}: {run(name, dict(params), dims, auto)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
