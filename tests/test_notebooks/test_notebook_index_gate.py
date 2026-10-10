"""The notebook index gate and the CI graph around it (H-0119 6., decision 1).

The index jobs run on every PR and every push to main; nothing selects them by
changed paths. The gate job always runs and passes only when the matrix job,
the registry probe and the notebook execution all succeeded. A failed,
cancelled or skipped job fails it, so no job can drop out silently.
"""

from __future__ import annotations

import importlib.util
import itertools
import pathlib
import sys
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
RESULTS = ("success", "failure", "cancelled", "skipped")
INDEX_JOBS = ("notebook-index-matrix", "extras-registry", "notebook-index-execution")
GATE = "notebook-index-gate"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "lizyml_notebook_index_gate", ROOT / "scripts" / "notebook_index_gate.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load()


@pytest.mark.parametrize("combination", list(itertools.product(RESULTS, repeat=3)))
def test_only_all_success_passes(combination: tuple[str, str, str]) -> None:
    ok, reason = gate.decide(*combination)
    assert ok is (combination == ("success", "success", "success")), reason
    assert reason


@pytest.mark.parametrize(
    "combination",
    [
        ("success", "success", "neutral"),  # an unknown result
        ("success", "", "success"),  # a missing result
        ("SUCCESS", "success", "success"),
    ],
)
def test_unknown_results_fail(combination: tuple[str, str, str]) -> None:
    assert gate.decide(*combination)[0] is False


def test_main_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("MATRIX_RESULT", "REGISTRY_RESULT", "EXECUTION_RESULT"):
        monkeypatch.setenv(key, "success")
    assert gate.main() == 0
    monkeypatch.setenv("EXECUTION_RESULT", "skipped")
    assert gate.main() == 1
    monkeypatch.delenv("MATRIX_RESULT")
    with pytest.raises(KeyError):
        gate.main()


# --- The workflow (acceptance criterion 6) -------------------------------------


def _workflow() -> dict[Any, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _triggers() -> dict[str, Any]:
    workflow = _workflow()
    # YAML 1.1 reads the bare key `on` as True.
    return workflow.get("on", workflow.get(True))


def test_the_workflow_runs_on_prs_and_pushes_to_main() -> None:
    triggers = _triggers()
    assert "pull_request" in triggers
    assert "main" in triggers["push"]["branches"]


@pytest.mark.parametrize("job", INDEX_JOBS)
def test_no_index_job_is_selected_by_a_condition(job: str) -> None:
    assert "if" not in _workflow()["jobs"][job]


def test_the_jobs_depend_on_the_matrix_and_the_gate_on_all_three() -> None:
    jobs = _workflow()["jobs"]
    assert jobs["extras-registry"]["needs"] == "notebook-index-matrix"
    assert jobs["notebook-index-execution"]["needs"] == "notebook-index-matrix"
    assert jobs[GATE]["needs"] == list(INDEX_JOBS)
    assert jobs[GATE]["if"] == "always()"


def test_the_gate_reads_each_job_result() -> None:
    (step,) = [s for s in _workflow()["jobs"][GATE]["steps"] if "env" in s]
    assert step["env"] == {
        "MATRIX_RESULT": "${{ needs.notebook-index-matrix.result }}",
        "REGISTRY_RESULT": "${{ needs.extras-registry.result }}",
        "EXECUTION_RESULT": "${{ needs.notebook-index-execution.result }}",
    }
    assert step["run"] == "python3 scripts/notebook_index_gate.py"


def test_the_scope_script_is_gone() -> None:
    assert not (ROOT / ".github" / "scripts" / "notebook_index_scope.sh").exists()
    assert "notebook_index_scope" not in WORKFLOW.read_text(encoding="utf-8")
