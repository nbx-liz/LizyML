"""Capture H-0149 sequences without training or depending on pytest.

From the repository root, pass all 16 interpreters as VERSION=PYTHON:
    python docs/audits/2026-11-i419/capture_sequences.py 3.1.0=/env/bin/python ...
Repeat that command to verify byte-identical output. No golden is written unless
all versions agree with the four expected bands. To check an installed version:
    /env/bin/python docs/audits/2026-11-i419/capture_sequences.py --check
The pytest module imports this driver's build_sequence and select_band by path.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "tests/test_tuning/data/h0149_tpe_sequences.json"
SEED = 42
N_TRIALS = 40
FLOAT_DIGITS = 12
EXPECTED_BANDS = (
    ("3.1", "3.2"),
    ("3.3", "3.4"),
    ("3.5", "3.6", "4.0", "4.1", "4.2", "4.3", "4.4", "4.5", "4.6", "4.7"),
    ("5.0",),
)
CAPTURE_VERSIONS = (
    "3.1.0",
    "3.2.0",
    "3.3.0",
    "3.4.0",
    "3.5.0",
    "3.6.0",
    "3.6.1",
    "4.0.0",
    "4.1.0",
    "4.2.0",
    "4.3.0",
    "4.4.0",
    "4.5.0",
    "4.6.0",
    "4.7.0",
    "5.0.0",
)


def build_sequence(sampler: Any | None = None) -> list[dict[str, Any]]:
    """Sample the binary default space; the objective uses unrounded values.

    Each numeric dimension is scaled to [0, 1] (in log space when declared).
    The objective is a weighted squared distance to fixed interior targets.
    Only the returned floats are rounded to twelve significant digits.
    """
    import optuna

    from lizyml.estimators.lgbm.defaults import default_space
    from lizyml.tuning.search_space import CategoricalDim, suggest_params
    from lizyml.tuning.tuner import make_tpe_sampler

    dims = default_space("binary")
    sequence: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        params = suggest_params(trial, dims)
        score = 0.0
        for index, dim in enumerate(dims):
            value = params[dim.name]
            if isinstance(dim, CategoricalDim):
                scaled = dim.choices.index(value) / max(1, len(dim.choices) - 1)
            elif dim.log:
                scaled = (math.log(value) - math.log(dim.low)) / (
                    math.log(dim.high) - math.log(dim.low)
                )
            else:
                scaled = (value - dim.low) / (dim.high - dim.low)
            target = (index + 1) / (len(dims) + 1)
            score += (index + 1) * (scaled - target) ** 2
        sequence.append(
            {
                key: float(format(value, f".{FLOAT_DIGITS}g"))
                if isinstance(value, float)
                else value
                for key, value in params.items()
            }
        )
        return score

    study = optuna.create_study(
        direction="minimize",
        sampler=make_tpe_sampler(SEED) if sampler is None else sampler,
    )
    study.optimize(objective, n_trials=N_TRIALS)
    return sequence


def select_band(data: dict[str, Any], version: str) -> dict[str, Any]:
    """Fail on an unknown minor, including a newly released Optuna minor."""
    match = re.match(r"^([0-9]+)\.([0-9]+)(?:\.|$)", version)
    minor = f"{int(match[1])}.{int(match[2])}" if match else None
    matches = [band for band in data["bands"] if minor in band["minors"]]
    if len(matches) != 1:
        raise AssertionError(
            f"Optuna {version} has no unique H-0149 golden band. Regenerate with "
            "docs/audits/2026-11-i419/capture_sequences.py VERSION=PYTHON ... "
            "for every minor in each band before adding the new minor."
        )
    return matches[0]


def capture_current() -> dict[str, Any]:
    import numpy
    import optuna

    return {
        "sequence": build_sequence(),
        "provenance": {
            "optuna": optuna.__version__,
            "numpy": numpy.__version__,
            "python": platform.python_version(),
        },
    }


def json_text(value: Any, level: int = 0) -> str:
    """Sorted, indented JSON with a fixed .12g float format and LF endings."""
    indent = "  " * level
    child_indent = "  " * (level + 1)
    if isinstance(value, dict):
        items = [
            f"{child_indent}{json.dumps(key)}: {json_text(value[key], level + 1)}"
            for key in sorted(value)
        ]
        return "{\n" + ",\n".join(items) + "\n" + indent + "}" if items else "{}"
    if isinstance(value, list):
        items = [child_indent + json_text(item, level + 1) for item in value]
        return "[\n" + ",\n".join(items) + "\n" + indent + "]" if items else "[]"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite float in H-0149 capture")
        return format(value, f".{FLOAT_DIGITS}g")
    return json.dumps(value, allow_nan=False)


def collect_environments(interpreters: list[str]) -> dict[str, dict[str, Any]]:
    results = {}
    for interpreter in interpreters:
        version, separator, python = interpreter.partition("=")
        if not separator or not python or version in results:
            raise ValueError(f"Expected a unique VERSION=PYTHON, got {interpreter!r}")
        process = subprocess.run(
            [python, str(Path(__file__).resolve()), "--emit"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        captured = json.loads(process.stdout)
        if captured["provenance"]["optuna"] != version:
            raise ValueError(f"{python} did not report Optuna {version}")
        results[version] = captured
    return results


def captured_data(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(results) != set(CAPTURE_VERSIONS):
        raise ValueError(f"Capture requires exactly these versions: {CAPTURE_VERSIONS}")
    groups: list[dict[str, Any]] = []
    for version in CAPTURE_VERSIONS:
        captured = results[version]
        minor = ".".join(version.split(".")[:2])
        group = next((g for g in groups if g["sequence"] == captured["sequence"]), None)
        if group is None:
            group = {"minors": [], "sequence": captured["sequence"], "provenance": []}
            groups.append(group)
        if minor not in group["minors"]:
            group["minors"].append(minor)
        group["provenance"].append(captured["provenance"])
    found = tuple(tuple(group["minors"]) for group in groups)
    if found != EXPECTED_BANDS:
        raise ValueError(f"STOP: unexpected H-0149 bands: {found!r}")
    return {
        "seed": SEED,
        "n_trials": N_TRIALS,
        "float_significant_digits": FLOAT_DIGITS,
        "bands": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("interpreters", nargs="*", metavar="VERSION=PYTHON")
    parser.add_argument("--check", action="store_true", help="Compare, never write")
    parser.add_argument("--emit", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=GOLDEN)
    args = parser.parse_args()
    if args.emit:
        print(json.dumps(capture_current(), sort_keys=True, allow_nan=False))
        return
    if args.check:
        data = json.loads(args.output.read_text(encoding="utf-8"))
        results = (
            collect_environments(args.interpreters)
            if args.interpreters
            else {"current": capture_current()}
        )
        for captured in results.values():
            version = captured["provenance"]["optuna"]
            band = select_band(data, version)
            if captured["sequence"] != band["sequence"]:
                raise AssertionError(f"Optuna {version} sequence differs from its band")
            print(f"Optuna {version}: 40 trials match band {', '.join(band['minors'])}")
        return
    data = captured_data(collect_environments(args.interpreters))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json_text(data) + "\n").encode("utf-8"))
    for band in data["bands"]:
        print(f"Band: {', '.join(band['minors'])} ({len(band['provenance'])} captures)")


if __name__ == "__main__":
    main()
