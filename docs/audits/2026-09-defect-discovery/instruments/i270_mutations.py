"""Mutation check for the #270 repairs.

For each mutation: apply it to a lizyml/ source file, run the repaired test
(must FAIL) and the pre-repair version of the same test taken from the base
commit (expected to PASS -- that is what made it hollow), then restore the
file byte-for-byte. Nothing is committed; every file is restored in `finally`.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
BASE = "1d41b662563a52c4c6aec1398f1a3abe5287f08f"
PY = str(ROOT / ".venv/bin/python")

MUTATIONS = [
    {
        "name": "filter_metrics keeps emptied branches",
        # The old test caught this shape too ({"oof": {}}); it was WEAK for an
        # emptied or missing branch, which the new assertions pin.
        "base_also_fails": True,
        "file": "lizyml/core/_model_metrics.py",
        "old": "        if _has_metric_content(filtered_top):\n            result[top_key] = filtered_top",
        "new": "        result[top_key] = filtered_top",
        "tests": ["tests/test_core/test_model_metrics_calibrated.py::TestFilterMetricsNoBranches::test_no_empty_calibrated_branch"],
    },
    {
        "name": "adapter drops metric and scale_pos_weight before lgb.train",
        "file": "lizyml/estimators/lgbm/adapter.py",
        "old": "            self._model = lgb.train(\n                params,",
        "new": "            self._model = lgb.train(\n                {k: v for k, v in params.items() if k not in ('metric', 'scale_pos_weight')},",
        "tests": [
            "tests/test_estimators/test_param_behavioral_effect.py::TestBoosterParamPropagation::test_metric_default_per_task",
            "tests/test_estimators/test_param_behavioral_effect.py::TestSmartParamsBehavior::test_balanced_binary_shifts_predictions",
        ],
    },
    {
        "name": "calibration params not canonicalised on the facade path",
        "file": "lizyml/core/_model_factories.py",
        "old": "        prepared = canonicalise_calibration_params(params)",
        "new": "        prepared = dict(params)",
        # The unit test beside it calls the canonicaliser directly, so this
        # mutation (on the facade's call) leaves it green by design.
        "tests": [
            "tests/test_core/test_fit_params_override.py::test_every_calibration_default_written_as_an_alias_reaches_training",
        ],
    },
    {
        "name": "one residuals figure skips the theme helper",
        "file": "lizyml/plots/residuals.py",
        "old": "    apply_default_layout(",
        "new": "    (lambda *a, **k: None)(",
        "tests": ["tests/test_plots/test_theme.py::TestThemeAppliedToAllPlots"],
        "all": True,
    },
    {
        "name": "Model.load fails",
        "file": "lizyml/core/_model_persistence.py",
        "old": "        fit_result, refit_result, metadata, analysis_context = _load(path)",
        "new": "        raise RuntimeError('mutant: load broken')",
        "tests": ["tests/test_estimators/test_lightgbm_parameter_names.py::test_loading_an_artifact_is_not_blocked_by_the_gate"],
    },
    {
        "name": "loader never unpickles fit_result.pkl",
        "file": "lizyml/persistence/loader.py",
        "old": "        fit_result = joblib.load(io.BytesIO(fit_raw))",
        "new": "        fit_result = None",
        "tests": ["tests/test_persistence/test_export_load_errors.py::TestLoadErrors::test_corrupt_fit_result_pkl"],
    },
    {
        "name": "duplicate-spelling gate runs before the value gate",
        "file": "lizyml/core/_model_factories.py",
        "old": "    normalised = normalise_params(params, surface=surface)\n    check_duplicate_identities(provider, normalised, surface=surface)",
        "new": "    check_duplicate_identities(provider, params, surface=surface)\n    normalised = normalise_params(params, surface=surface)",
        "tests": ["tests/test_core/test_fit_params_override.py::test_a_hostile_value_is_refused_beside_a_second_spelling_too"],
    },
]


def run(ids: list[str]) -> tuple[str, str]:
    """Run pytest; return ("passed" | "failed" | "broken", the summary line).

    "failed" requires pytest's tests-failed exit code and no errors, so a
    collection, fixture or start-up error is never read as a killed mutation.
    """
    proc = subprocess.run(  # noqa: S603
        [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-cov", *ids],
        cwd=ROOT, capture_output=True, text=True, timeout=900,
    )
    lines = [l for l in proc.stdout.splitlines() if l.strip()]
    tail = lines[-1] if lines else proc.stderr.strip()[-300:]
    if proc.returncode == 0 and "passed" in tail:
        return "passed", tail
    if proc.returncode == 1 and "failed" in tail and "error" not in tail:
        return "failed", tail
    return "broken", f"rc={proc.returncode}: {tail}"


def old_copy(test_id: str) -> tuple[pathlib.Path, str]:
    """Write the base commit's version of the test module beside it; return its id."""
    path, _, rest = test_id.partition("::")
    src = subprocess.run(["git", "show", f"{BASE}:{path}"], cwd=ROOT,  # noqa: S603, S607
                         capture_output=True, text=True, check=True).stdout
    tmp = ROOT / pathlib.Path(path).with_name("test_zz_i270_base_" + pathlib.Path(path).name)
    tmp.write_text(src, encoding="utf-8")
    return tmp, f"{tmp.relative_to(ROOT).as_posix()}::{rest}"


OLD_NAME = {
    # repaired tests whose name changed or which are new; map to the base name
    "test_every_calibration_default_written_as_an_alias_reaches_training": None,
    "TestThemeAppliedToAllPlots": "TestThemeAppliedToAllPlots::test_every_plot_module_imports_apply_default_layout",
}

failures = 0
ONLY = sys.argv[1] if len(sys.argv) > 1 else None
for m in MUTATIONS:
    if ONLY and ONLY not in m["name"]:
        continue
    f = ROOT / m["file"]
    original = f.read_bytes()
    text = original.decode()
    count = text.count(m["old"])
    if count < 1:
        print(f"!! {m['name']}: mutation site not found")
        failures += 1
        continue
    temps: list[pathlib.Path] = []
    print(f"== {m['name']}  ({m['file']}, {'every site' if m.get('all') else 'site 1'} of {count})")
    # Each repaired test must pass before the mutation, or its failure under
    # the mutation says nothing about the mutation.
    for t in m["tests"]:
        state, tail = run([t])
        if state != "passed":
            failures += 1
            print(f"   unmutated: {tail}   <- REPAIRED TEST DOES NOT PASS")
    try:
        mutated = (text.replace(m["old"], m["new"]) if m.get("all")
                   else text.replace(m["old"], m["new"], 1))
        f.write_text(mutated, encoding="utf-8")
        for t in m["tests"]:
            state, tail = run([t])
            ok = state == "failed"
            failures += 0 if ok else 1
            print(f"   repaired: {tail}   <- {'OK (fails)' if ok else 'NOT KILLED'}")
            leaf = t.split("::", 1)[1]
            base_leaf = OLD_NAME.get(leaf.split("::")[-1], OLD_NAME.get(leaf, leaf))
            if base_leaf is None:
                print("   base    : (new test, no base version)")
                continue
            tmp, old_id = old_copy(t.split("::")[0] + "::" + base_leaf)
            temps.append(tmp)
            base_state, base_tail = run([old_id])
            expected = "failed" if m.get("base_also_fails") else "passed"
            base_ok = base_state == expected
            failures += 0 if base_ok else 1
            print(f"   base    : {base_tail}   <- "
                  f"{'as expected' if base_ok else f'EXPECTED {expected.upper()}'}")
    finally:
        f.write_bytes(original)
        for tmp in temps:
            tmp.unlink(missing_ok=True)
print("all mutations killed" if failures == 0 else f"{failures} mutation(s) survived or missing")
sys.exit(1 if failures else 0)
