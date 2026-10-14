"""Assert the environment holds exactly the extras a notebook declares (H-0119 6.(b)).

Run by CI before the index execution, in the per-notebook environment::

    uv run --no-sync --no-dev python -m \
        tests.test_notebooks.check_installed_extras <stem>

For each extra in ``lizyml/_extras.py``, the package it installs must import
when the notebook declares the extra and must be absent when it does not. The
same helper checks the environments of the registry probe (6.(a)).

It is not a pytest test: in the full dev environment every extra is
installed, so it would fail there by design.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Iterable, Sequence

from tests.test_notebooks.index_runner import ROOT, load_examples_index


def assert_installed(present: Iterable[str], absent: Iterable[str]) -> list[str]:
    """Import every ``present`` extra's package; find none of the ``absent`` ones."""
    packages = load_examples_index().REGISTRY.EXTRA_PACKAGES
    present, absent = sorted(present), sorted(absent)
    if set(present) | set(absent) != set(packages) or set(present) & set(absent):
        raise SystemExit(
            f"present {present} and absent {absent} must partition {sorted(packages)}"
        )
    report: list[str] = []
    for extra in present:
        module = importlib.import_module(packages[extra])
        report.append(
            f"{extra}: {packages[extra]} {getattr(module, '__version__', '?')} present"
        )
    for extra in absent:
        if importlib.util.find_spec(packages[extra]) is not None:
            raise SystemExit(
                f"{extra}: {packages[extra]} is installed but must be absent"
            )
        report.append(f"{extra}: {packages[extra]} absent")
    return report


def main(argv: Sequence[str]) -> int:
    if len(argv) != 1:
        raise SystemExit("usage: check_installed_extras <notebook stem>")
    ix = load_examples_index()
    path = ROOT / "notebooks" / f"{argv[0]}.ipynb"
    declaration = ix.check_notebook(ix.read_notebook(path))
    declared = set(declaration.extras)
    for line in assert_installed(declared, set(ix.REGISTRY.EXTRA_PACKAGES) - declared):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
