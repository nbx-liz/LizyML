"""Each notebook runs its index examples on a real ``Model`` (H-0119 6.(b)).

CI runs one notebook per job, in an environment holding only the extras the
notebook declares::

    uv run --no-sync --no-dev python -m pytest \\
        tests/test_notebooks/test_index_execution.py -m slow -k <notebook stem>

A separate CI step before pytest (``check_installed_extras.py``) asserts that
the environment holds exactly those extras; that check is not in this test,
because the main-branch quality lane also runs it in the full dev environment,
where it checks the recording only.

The notebook is instrumented in memory (``index_runner.py``): every tagged
statement must run, on the declared receiver, as an outermost ``Model`` call. A
missing extra fails the execution. A network failure is retried (three
attempts in all, each with a new kernel and working directory) and then fails;
it is never skipped.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from tests.test_notebooks import index_runner

pytestmark = pytest.mark.slow

PATHS = sorted(index_runner.NOTEBOOKS.glob("*.ipynb"))


def test_the_parametrization_has_its_anchor() -> None:
    assert len(PATHS) == 8


@pytest.mark.parametrize("path", PATHS, ids=[p.stem for p in PATHS])
def test_notebook_runs_its_index_examples(
    path: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    log = index_runner.run_instrumented(nb, workdir_base=tmp_path)
    assert 1 <= len(log) <= index_runner.ATTEMPTS
