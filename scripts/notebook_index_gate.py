"""Decide the notebook index gate from the CI job results (H-0119 6., decision 1).

The ``notebook-index-gate`` job runs with ``if: always()`` after the matrix
job and the two substantive jobs (the registry probe and the notebook
execution), and is the single check to require. The index jobs run whenever
ci.yml runs (every PR to main or develop and every push to main), so the gate
passes only when all three succeeded. Anything
else fails: a failed, cancelled or skipped job, or a result outside the known
ones.

Usage (standard library only)::

    MATRIX_RESULT=... REGISTRY_RESULT=... EXECUTION_RESULT=... \\
        python3 scripts/notebook_index_gate.py
"""

from __future__ import annotations

import os
import sys


def decide(matrix: str, registry: str, execution: str) -> tuple[bool, str]:
    """Whether the gate passes, and why."""
    observed = (
        f"notebook-index-matrix={matrix} extras-registry={registry} "
        f"notebook-index-execution={execution}"
    )
    if matrix == registry == execution == "success":
        return True, f"the index jobs ran and passed ({observed})"
    return False, f"an index job did not succeed ({observed})"


def main() -> int:
    ok, reason = decide(
        os.environ["MATRIX_RESULT"],
        os.environ["REGISTRY_RESULT"],
        os.environ["EXECUTION_RESULT"],
    )
    print(("pass: " if ok else "FAIL: ") + reason)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
