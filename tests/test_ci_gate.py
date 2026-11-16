"""The CI gate passes only when every blocking job succeeded (H-0146 acceptance 2).

``scripts/ci_gate.py`` decides the ``ci-gate`` job of ``ci.yml`` from
``toJSON(needs)``. Its decision is tested here on every result GitHub reports,
and the workflow is read to check that every job of ``ci.yml`` is either needed
by the gate or on the script's closed exclusion list, so a new job cannot be
left out of the required check unnoticed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ci_gate.py"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("lizyml_ci_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ci_gate = _load()


def _needs(**overrides: str) -> dict[str, Any]:
    results = {job: "success" for job in ci_gate.REQUIRED_JOBS}
    results.update(overrides)
    return {job: {"result": result, "outputs": {}} for job, result in results.items()}


def test_every_job_successful_passes() -> None:
    ok, reason = ci_gate.decide(json.dumps(_needs()))
    assert ok, reason


@pytest.mark.parametrize("job", ci_gate.REQUIRED_JOBS)
@pytest.mark.parametrize(
    "result", ["failure", "cancelled", "skipped", "neutral", "", "SUCCESS"]
)
def test_any_job_not_successful_fails(job: str, result: str) -> None:
    ok, reason = ci_gate.decide(json.dumps(_needs(**{job: result})))
    assert not ok
    assert job in reason


@pytest.mark.parametrize("job", ci_gate.REQUIRED_JOBS)
def test_a_missing_job_fails(job: str) -> None:
    needs = _needs()
    del needs[job]
    ok, reason = ci_gate.decide(json.dumps(needs))
    assert not ok
    assert f"{job}=<missing>" in reason


@pytest.mark.parametrize(
    "entry",
    [None, "success", {"outputs": {}}, {"result": None}, {"result": 1}],
)
def test_a_malformed_entry_fails(entry: object) -> None:
    needs: dict[str, object] = dict(_needs())
    needs["quality"] = entry
    ok, _ = ci_gate.decide(json.dumps(needs))
    assert not ok


def test_an_unexpected_job_fails() -> None:
    needs = _needs()
    needs["quality-latest-deps"] = {"result": "success", "outputs": {}}
    ok, reason = ci_gate.decide(json.dumps(needs))
    assert not ok
    assert "quality-latest-deps" in reason


@pytest.mark.parametrize("raw", ["", "not json", "[]", "null", '"success"'])
def test_input_that_is_not_the_needs_object_fails(raw: str) -> None:
    ok, _ = ci_gate.decide(raw)
    assert not ok


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    base = {k: v for k, v in os.environ.items() if k != "NEEDS_JSON"}
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        env={**base, **env},
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_script_exits_zero_only_on_a_pass() -> None:
    passed = _run({"NEEDS_JSON": json.dumps(_needs())})
    assert passed.returncode == 0
    assert passed.stdout.startswith("pass: ")
    failed = _run({"NEEDS_JSON": json.dumps(_needs(quality="failure"))})
    assert failed.returncode == 1
    assert failed.stdout.startswith("FAIL: ")
    unset = _run({})
    assert unset.returncode == 1


# ---------------------------------------------------------------------------
# The workflow wires the gate to exactly the blocking jobs
# ---------------------------------------------------------------------------


def _jobs(text: str | None = None) -> dict[str, dict[str, Any]]:
    source = WORKFLOW.read_text(encoding="utf-8") if text is None else text
    jobs: dict[str, dict[str, Any]] = yaml.safe_load(source)["jobs"]
    return jobs


def _wiring_problems(jobs: dict[str, dict[str, Any]]) -> list[str]:
    gate = jobs["ci-gate"]
    needs = gate["needs"]
    problems = []
    if list(needs) != list(ci_gate.REQUIRED_JOBS):
        problems.append(f"needs {needs} != REQUIRED_JOBS {list(ci_gate.REQUIRED_JOBS)}")
    if gate.get("if") != "always()":
        problems.append(f"if is {gate.get('if')!r}, not always()")
    others = set(jobs) - {"ci-gate"}
    unaccounted = others - set(needs) - set(ci_gate.EXCLUDED_JOBS)
    if unaccounted:
        problems.append(f"jobs neither needed nor excluded: {sorted(unaccounted)}")
    both = set(needs) & set(ci_gate.EXCLUDED_JOBS)
    if both:
        problems.append(f"jobs both needed and excluded: {sorted(both)}")
    stale = set(ci_gate.EXCLUDED_JOBS) - others
    if stale:
        problems.append(f"excluded jobs that do not exist: {sorted(stale)}")
    return problems


def test_every_job_is_needed_by_the_gate_or_excluded_with_a_reason() -> None:
    assert _wiring_problems(_jobs()) == []
    assert all(reason for reason in ci_gate.EXCLUDED_JOBS.values())


def test_the_exclusion_list_is_closed() -> None:
    assert set(ci_gate.EXCLUDED_JOBS) == {
        "quality-latest-deps",
        "notebook-index-matrix",
        "extras-registry",
        "notebook-index-execution",
    }


def test_a_new_job_left_out_of_the_gate_is_caught() -> None:
    # Negative control: the wiring check must fail on a job nobody accounted for.
    jobs = _jobs()
    jobs["new-job"] = {"runs-on": "ubuntu-latest", "steps": []}
    assert any("new-job" in p for p in _wiring_problems(jobs))


def test_the_gate_reads_the_needs_through_the_environment() -> None:
    steps = _jobs()["ci-gate"]["steps"]
    decide = steps[-1]
    assert decide["run"] == "python3 scripts/ci_gate.py"
    assert decide["env"] == {"NEEDS_JSON": "${{ toJSON(needs) }}"}
