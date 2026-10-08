"""Run each repaired boundary test under the kill mode that made it hollow."""

import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
INS = pathlib.Path(__file__).resolve().parent
W = INS.parent / "results" / "i270"
modes = {x["test"]: x["mode"] for x in json.loads((W / "confirmed.json").read_text())}

# base key -> node id(s) at the repaired head (renames and splits mapped)
REPAIRED = {
    "tests/test_core/test_contracts.py::TestFitResultSchema::test_oof_pred_is_ndarray": None,
    "tests/test_core/test_contracts.py::TestFitResultSchema::test_if_pred_per_fold_len_equals_n_splits": None,
    "tests/test_core/test_contracts.py::TestFitResultSchema::test_metrics_raw_structure": None,
    "tests/test_core/test_contracts.py::TestFitResultSchema::test_calibrated_key_absent_when_no_calibrator": None,
    "tests/test_estimators/test_param_behavioral_effect.py::TestBoosterParamPropagation::test_metric_default_per_task": None,
    "tests/test_estimators/test_param_behavioral_effect.py::TestSmartParamsBehavior::test_balanced_binary_shifts_predictions": None,
    "tests/test_estimators/test_lightgbm_parameter_names.py::test_loading_an_artifact_is_not_blocked_by_the_gate": None,
    "tests/test_plots/test_theme.py::TestThemeAppliedToAllPlots::test_every_plot_module_imports_apply_default_layout":
        "tests/test_plots/test_theme.py::TestThemeAppliedToAllPlots::test_every_public_plot_figure_passes_through_the_helper",
    "tests/test_core/test_fit_params_override.py::test_every_calibration_alias_is_canonical_before_the_defaults_merge":
        "tests/test_core/test_fit_params_override.py::test_every_calibration_default_written_as_an_alias_reaches_training",
}
#: The noun classifier mapped "metric" to the metric producers, but this claim
#: (which metrics LightGBM evaluates by default) is produced by lightgbm.train.
MODE_OVERRIDE = {
    "tests/test_estimators/test_param_behavioral_effect.py::TestBoosterParamPropagation::test_metric_default_per_task": "train",
}

failures = 0
for base, now in REPAIRED.items():
    mode = MODE_OVERRIDE.get(base, modes[base])
    node = now or base
    env = {**os.environ, "LIZYML_KILL": mode, "PYTHONPATH": str(INS)}
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
           "--no-cov", "-p", "kill_producers", node]
    # Unkilled first: the test must pass, or a failure under the kill says
    # nothing about the producer.
    plain = subprocess.run(  # noqa: S603
        cmd, cwd=ROOT, capture_output=True, text=True, timeout=600,
        env={**env, "LIZYML_KILL": ""},
    )
    proc = subprocess.run(  # noqa: S603
        cmd, cwd=ROOT, capture_output=True, text=True, env=env, timeout=600,
    )
    out = proc.stdout + proc.stderr
    lines = [l for l in proc.stdout.splitlines() if l.strip()]
    tail = lines[-1] if lines else f"no output; stderr: {proc.stderr.strip()[-200:]}"
    armed = f"[kill_producers] MODE={mode!r}: disabled" in out
    killed = plain.returncode == 0 and proc.returncode == 1 and armed and "ProducerRan" in out
    failures += 0 if killed else 1
    if plain.returncode != 0:
        verdict = "DOES NOT PASS UNKILLED"
    else:
        verdict = "fails with ProducerRan" if killed else "STILL HOLLOW OR BROKEN"
    print(f"{mode:6s} {verdict:22s} {tail:40s} {node.split('::')[-1]}")
raise SystemExit(1 if failures else 0)
