"""Decide the notebook index gate from the CI job results (H-0119 6.).

The ``notebook-index-gate`` job runs with ``if: always()`` after the scope job
and the two substantive jobs (the registry probe and the notebook execution),
and is the single check to require. It passes in exactly two cases:

- the scope job succeeded with ``run=true`` and both substantive jobs
  succeeded;
- the scope job succeeded with ``run=false`` and both substantive jobs were
  skipped.

Any other combination fails: a failed or cancelled job, a skipped substantive
job when ``run=true``, a substantive job that ran when ``run=false``, a scope
job that did not succeed (its dependents are then skipped, which must not read
as "nothing to check"), and any value outside the known ones.

Usage (standard library only)::

    SCOPE_RESULT=... SCOPE_RUN=... REGISTRY_RESULT=... EXECUTION_RESULT=... \\
        python3 scripts/notebook_index_gate.py
"""

from __future__ import annotations

import os
import sys


def decide(scope: str, run: str, registry: str, execution: str) -> tuple[bool, str]:
    """Whether the gate passes, and why."""
    observed = (
        f"scope={scope} run={run!r} extras-registry={registry} "
        f"notebook-index-execution={execution}"
    )
    if scope != "success":
        return False, f"the scope job did not succeed ({observed})"
    if run == "true" and registry == execution == "success":
        return True, f"the index jobs ran and passed ({observed})"
    if run == "false" and registry == execution == "skipped":
        return True, f"no index path changed; the jobs were skipped ({observed})"
    return False, f"inconsistent or failed index jobs ({observed})"


def main() -> int:
    ok, reason = decide(
        os.environ["SCOPE_RESULT"],
        os.environ["SCOPE_RUN"],
        os.environ["REGISTRY_RESULT"],
        os.environ["EXECUTION_RESULT"],
    )
    print(("pass: " if ok else "FAIL: ") + reason)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
