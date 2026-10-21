"""H-0140 acceptance 10: every Linux pytest step of CI runs under the memory ceiling.

A step that can run on Linux (a job on ``ubuntu-*``, or a matrix job whose
step is not limited to another OS) and runs ``pytest`` must start with
``ulimit -v "$PYTEST_MEMORY_CEILING_KIB"``. The count is pinned, so a new
pytest step has to be classified here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
CEILING = 'ulimit -v "$PYTEST_MEMORY_CEILING_KIB"'
LINUX_PYTEST_STEPS = 5


def _can_run_on_linux(job: dict[str, Any], step: dict[str, Any]) -> bool:
    runs_on = str(job.get("runs-on", ""))
    condition = str(step.get("if", ""))
    if "runner.os != 'Linux'" in condition:
        return False
    if "runner.os == 'Linux'" in condition:
        return True
    if "matrix.os" in runs_on:
        oses = job.get("strategy", {}).get("matrix", {}).get("os", [])
        return any(str(os).startswith("ubuntu") for os in oses)
    return runs_on.startswith("ubuntu")


def _linux_pytest_steps() -> list[tuple[str, dict[str, Any]]]:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    found = []
    for name, job in workflow["jobs"].items():
        for step in job.get("steps", []):
            if "pytest" in str(step.get("run", "")) and _can_run_on_linux(job, step):
                found.append((f"{name}: {step.get('name', '?')}", step))
    return found


def test_the_ceiling_is_declared_once() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert int(workflow["env"]["PYTEST_MEMORY_CEILING_KIB"]) > 0


def test_every_linux_pytest_step_runs_under_the_ceiling() -> None:
    steps = _linux_pytest_steps()
    assert len(steps) == LINUX_PYTEST_STEPS, [name for name, _ in steps]
    missing = [
        name
        for name, step in steps
        if not str(step["run"]).lstrip().startswith(CEILING)
    ]
    assert missing == []
