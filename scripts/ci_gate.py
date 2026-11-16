"""Decide the CI gate from the results of the jobs it needs (H-0146).

The ``ci-gate`` job of ``.github/workflows/ci.yml`` runs with ``if: always()``
after every blocking job and is the single CI check the rulesets require. It
passes only when every needed job finished with ``success``. Anything else
fails: a failed, cancelled or skipped job, a result outside the known ones, a
needed job missing from the input, or input that is not the expected JSON.

The input is ``toJSON(needs)``, passed through the environment::

    NEEDS_JSON='{"quality": {"result": "success", "outputs": {}}, ...}' \\
        python3 scripts/ci_gate.py

Standard library only: the job runs it without installing anything.
"""

from __future__ import annotations

import json
import os
import sys

#: The jobs ``ci-gate`` needs, exactly as ``needs:`` lists them in ci.yml.
#: tests/test_ci_gate.py checks the two lists are the same.
REQUIRED_JOBS: tuple[str, ...] = (
    "git-policy",
    "quality",
    "quality-lowest-deps",
    "smoke-os-matrix",
    "distribution",
    "notebook-index-gate",
    "workflow-lint",
)

#: Jobs of ci.yml that ``ci-gate`` deliberately does not need, with the reason.
EXCLUDED_JOBS: dict[str, str] = {
    "quality-latest-deps": "non-blocking forward-compat lane (continue-on-error, #180)",
    "notebook-index-matrix": "collected by notebook-index-gate (H-0119)",
    "extras-registry": "collected by notebook-index-gate (H-0119)",
    "notebook-index-execution": "collected by notebook-index-gate (H-0119)",
}


def decide(needs_json: str) -> tuple[bool, str]:
    """Whether the gate passes, and why."""
    try:
        needs = json.loads(needs_json)
    except json.JSONDecodeError as exc:
        return False, f"NEEDS_JSON is not JSON ({exc})"
    if not isinstance(needs, dict):
        return False, f"NEEDS_JSON is not an object ({type(needs).__name__})"

    results: dict[str, str] = {}
    for job in REQUIRED_JOBS:
        entry = needs.get(job)
        result = entry.get("result") if isinstance(entry, dict) else None
        results[job] = result if isinstance(result, str) else "<missing>"
    unexpected = sorted(set(needs) - set(REQUIRED_JOBS))

    observed = " ".join(f"{job}={result}" for job, result in results.items())
    if unexpected:
        return False, f"unexpected jobs in needs: {', '.join(unexpected)} ({observed})"
    failed = [job for job, result in results.items() if result != "success"]
    if failed:
        return False, f"not successful: {', '.join(failed)} ({observed})"
    return True, f"every needed job succeeded ({observed})"


def main() -> int:
    ok, reason = decide(os.environ.get("NEEDS_JSON", ""))
    print(("pass: " if ok else "FAIL: ") + reason)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
