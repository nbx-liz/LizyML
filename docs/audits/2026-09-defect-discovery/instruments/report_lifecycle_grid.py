"""Execute H-0094 decision 13's declared set: values x lifecycles x surfaces.

Decision 13 says the reporting surfaces answer for the fit that happened, and it
names the set that claim quantifies over: the four values `export_code` and
`params_table` derive from something other than the trained parameter dict, under
five lifecycles. The rounds 15-16 monitor asked for that declaration to be held to
execution rather than to a table -- **40 cells** -- because decision 12 made the
same kind of claim over one lifecycle and round 16 falsified it.

Each cell compares one **surface** reading against the **ground truth**: what the
training actually consumed, captured while it ran rather than read back from a
config. A cell is `n/a` only with a stated reason; `params_table` has no seed row,
which is a gap in the table and not a disagreement.

Run:

    uv run python docs/audits/2026-09-defect-discovery/instruments/report_lifecycle_grid.py

Exits non-zero when any executed cell disagrees, so the declaration cannot go
stale in silence.
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any
from unittest import mock

import lizyml.core.model as model_mod
from lizyml import Model

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests._helpers import make_binary_df, make_config  # noqa: E402

warnings.simplefilter("ignore")

#: A study that moves every tunable member of the set away from its config value,
#: so a reader taking the wrong source is visible rather than accidentally right.
SPACE: dict[str, Any] = {
    "early_stopping_rounds": {
        "type": "categorical",
        "choices": [2],
        "category": "training",
    },
    "validation_ratio": {
        "type": "categorical",
        "choices": [0.45],
        "category": "training",
    },
}

CONFIG_PATIENCE = 7
CONFIG_RATIO = 0.2
CONFIG_SEED = 0
CONFIG_ROUNDS = 10

VALUES = ("early_stopping_rounds", "validation_ratio", "seed", "num_boost_round")
#: Decision 13's five, plus one. The sixth is the lifecycle issue #281 names,
#: and it is here because without it the `known-bound` branch below is a branch
#: no input reaches -- the DC6 shape, shipped inside the instrument whose whole
#: job is to catch declarations nothing executes. Decision 13's own set does not
#: contain the lifecycle its stated bound bites in.
LIFECYCLES = (
    "fit",
    "tune_then_fit",
    "fit_then_tune",
    "fit_export_load",
    "fit_tune_export_load",
    "tune_fit_export_load",
)
DECLARED_LIFECYCLES = 5
SURFACES = ("params_table", "export_code")

#: Cells with no reading to compare, each with the reason it has none.
NOT_APPLICABLE: dict[tuple[str, str], str] = {
    ("seed", "params_table"): (
        "params_table has no seed row: params_summary lists booster params by "
        "identity and seed is not among the names it reports"
    ),
}


def build() -> Model:
    cfg = make_config(
        "binary",
        n_estimators=CONFIG_ROUNDS,
        n_splits=2,
        num_threads=1,
        tuning_n_trials=1,
    )
    cfg["training"]["early_stopping"] = {
        "enabled": True,
        "rounds": CONFIG_PATIENCE,
        "validation_ratio": CONFIG_RATIO,
    }
    cfg["tuning"]["optuna"]["space"] = dict(SPACE)
    return Model(cfg, data=make_binary_df(n=200))


class _TrainingTruth:
    """What the training actually consumed, captured while it ran."""

    def __init__(self) -> None:
        self.ratios: list[float] = []
        self.seeds: list[Any] = []
        self.rounds: list[int] = []
        self.patience: Any = None


def _fit_capturing(model: Model) -> _TrainingTruth:
    """Fit, recording the inner-valid ratio and what reached ``lgb.train``."""
    import lightgbm as lgb

    truth = _TrainingTruth()
    real_factory = model_mod.make_inner_valid_factory
    real_build = model_mod.build_inner_valid
    real_train = lgb.train

    def spy_factory(cfg: Any) -> Any:
        inner = real_factory(cfg)

        def wrapped(ratio: float) -> Any:
            truth.ratios.append(float(ratio))
            return inner(ratio)

        return wrapped

    def spy_build(cfg: Any) -> Any:
        # The other branch. It resolves the ratio itself, so the configured
        # value is what it uses -- recorded from the same place the branch
        # reads it, not assumed.
        strategy = real_build(cfg)
        truth.ratios.append(float(getattr(strategy, "ratio", 0.0)))
        return strategy

    def spy_train(params: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        truth.seeds.append(params.get("seed"))
        rounds = kwargs.get("num_boost_round")
        if rounds is None and len(args) >= 2:
            rounds = args[1]
        truth.rounds.append(int(rounds))
        return real_train(params, *args, **kwargs)

    with (
        mock.patch.object(model_mod, "make_inner_valid_factory", spy_factory),
        mock.patch.object(model_mod, "build_inner_valid", spy_build),
        mock.patch.object(lgb, "train", spy_train),
    ):
        model.fit()

    truth.patience = model.fit_result.models[0].early_stopping_rounds
    return truth


def _reported(model: Model) -> dict[str, Any]:
    table = model.params_table()
    index = set(table.index)
    return {
        "early_stopping_rounds": table.loc["early_stopping_rounds", "value"],
        "validation_ratio": table.loc["validation_ratio", "value"],
        "seed": None,
        "num_boost_round": (
            table.loc["num_iterations", "value"] if "num_iterations" in index else None
        ),
    }


def _exported(model: Model) -> dict[str, Any]:
    with mock.patch("lizyml.codegen.generator.generate_code") as generate:
        model.export_code("not-written")
    kwargs = generate.call_args.kwargs
    return {
        "early_stopping_rounds": kwargs["early_stopping_rounds"],
        "validation_ratio": kwargs["validation_ratio"],
        "seed": kwargs["seed"],
        "num_boost_round": kwargs["num_boost_round"],
    }


def _run_lifecycle(name: str, workdir: Path) -> tuple[Model, dict[str, Any]]:
    """Return the model to report from, and the ground truth for its fit."""
    model = build()

    if name in ("tune_then_fit", "tune_fit_export_load"):
        model.tune()
        truth = _fit_capturing(model)
    else:
        truth = _fit_capturing(model)
        if name in ("fit_then_tune", "fit_tune_export_load"):
            model.tune()

    ground = {
        "early_stopping_rounds": truth.patience,
        # One distinct ratio must have been used, or the capture is ambiguous
        # and the cell cannot claim anything.
        "validation_ratio": (
            sorted(set(truth.ratios))[0] if len(set(truth.ratios)) == 1 else None
        ),
        "seed": sorted(set(truth.seeds))[0] if len(set(truth.seeds)) == 1 else None,
        "num_boost_round": (
            sorted(set(truth.rounds))[0] if len(set(truth.rounds)) == 1 else None
        ),
    }

    if name in ("fit_export_load", "fit_tune_export_load", "tune_fit_export_load"):
        path = workdir / name
        model.export(path)
        # The ground truth is still the original fit's: loading does not train.
        model = Model.load(path)

    return model, ground


def main() -> int:
    rows: list[tuple[str, str, str, Any, Any, str]] = []

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for lifecycle in LIFECYCLES:
            model, ground = _run_lifecycle(lifecycle, workdir)
            readings = {
                "params_table": _reported(model),
                "export_code": _exported(model),
            }
            for value in VALUES:
                for surface in SURFACES:
                    reason = NOT_APPLICABLE.get((value, surface))
                    if reason is not None:
                        rows.append(
                            (lifecycle, value, surface, ground[value], None, "n/a")
                        )
                        continue
                    expected = ground[value]
                    actual = readings[surface][value]
                    if expected is None:
                        verdict = "unmeasured"
                    elif float(expected) == float(actual):
                        verdict = "agrees"
                    elif (
                        value == "validation_ratio"
                        and lifecycle == "tune_fit_export_load"
                    ):
                        # The bound decision 13 states and #281 tracks: nothing
                        # in the artifact records which fit consumed the
                        # overlay, so a loaded model falls back to the config.
                        # This is the only cell it bites in -- and it is not in
                        # decision 13's own five lifecycles, which is why the
                        # sixth is here.
                        verdict = "known-bound"
                    else:
                        verdict = "DISAGREES"
                    rows.append(
                        (lifecycle, value, surface, expected, actual, verdict)
                    )

    width = max(len(lifecycle) for lifecycle in LIFECYCLES)
    print(f"{'lifecycle':<{width}}  {'value':<21} {'surface':<13} "
          f"{'used':>8} {'reported':>9}  verdict")
    for lifecycle, value, surface, expected, actual, verdict in rows:
        print(
            f"{lifecycle:<{width}}  {value:<21} {surface:<13} "
            f"{str(expected):>8} {str(actual):>9}  {verdict}"
        )

    counts: dict[str, int] = {}
    for *_, verdict in rows:
        counts[verdict] = counts.get(verdict, 0) + 1

    expected_cells = len(VALUES) * len(LIFECYCLES) * len(SURFACES)
    declared_cells = len(VALUES) * DECLARED_LIFECYCLES * len(SURFACES)
    print()
    print(
        f"cells: {len(rows)} "
        f"({declared_cells} from decision 13, "
        f"{expected_cells - declared_cells} for the bound issue #281 names)"
    )
    for verdict in sorted(counts):
        print(f"  {verdict}: {counts[verdict]}")
    for (value, surface), reason in NOT_APPLICABLE.items():
        print(f"  n/a {value} x {surface}: {reason}")

    assert len(rows) == expected_cells, (len(rows), expected_cells)
    # A `known-bound` cell that no input reaches would be a declaration nothing
    # executes, inside the instrument that exists to catch exactly that.
    assert counts.get("known-bound", 0) == len(SURFACES), counts
    failures = counts.get("DISAGREES", 0) + counts.get("unmeasured", 0)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
