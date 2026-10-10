"""Execute a notebook and record that its index examples ran (H-0119 6.(b)).

The notebook is rewritten in memory only:

- a first cell loads ``index_recorder.py`` and wraps ``Model``;
- each ``index-example`` statement is preceded by ``expect(R, "m", key)``
  (``R`` is a ``Model``) and followed by ``confirm(key)`` (that call ran,
  outermost);
- a last cell checks that every tagged statement was confirmed and prints a
  sentinel, which the runner requires in the executed notebook.

Execution uses ipykernel's native kernel started with this process's
``sys.executable``. Each attempt gets a new kernel and a fresh copy of
``notebooks/`` as its working directory; only a ``CellExecutionError`` carrying a
network marker is retried, at most three attempts in all, and exhausting them
fails (never skips).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import nbformat
from ipykernel.kernelspec import get_kernel_dict
from jupyter_client.kernelspec import KernelSpec, KernelSpecManager
from jupyter_client.manager import AsyncKernelManager
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from tests.test_notebooks.index_recorder import SENTINEL
from tests.test_notebooks.network_markers import NETWORK_ERROR_MARKERS

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "notebooks"
RECORDER_FILE = Path(__file__).with_name("index_recorder.py")
RECORDER_NAME = "_lizyml_index_recorder"
ATTEMPTS = 3
CELL_TIMEOUT = 600

__all__ = ["SENTINEL", "run_with_retries", "instrument_notebook", "run_instrumented"]


def load_examples_index() -> ModuleType:
    """``scripts/examples_index.py`` (the static checks and the grammar)."""
    name = "lizyml_examples_index"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / "examples_index.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --- instrumentation -------------------------------------------------------------


def _code_cell(source: str, nb: dict[str, Any], cell_id: str) -> dict[str, Any]:
    cell: dict[str, Any] = {
        "cell_type": "code",
        "metadata": {},
        "source": source,
        "outputs": [],
        "execution_count": None,
    }
    if nb.get("nbformat_minor", 0) >= 5:
        cell["id"] = cell_id
    return cell


def instrument_notebook(nb: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``nb`` with the recording cells and guards added.

    The notebook must pass the static contract first (``check_notebook``).
    """
    ix = load_examples_index()
    declaration = ix.check_notebook(nb)
    statements = ix.tagged_statements(nb, declaration)
    out = copy.deepcopy(nb)
    keys: list[str] = []
    guarded: dict[int, list[str]] = {}
    for s in statements:
        key = f"{s.cell}:{s.number}"
        keys.append(key)
        guarded.setdefault(s.cell, []).extend(
            [
                f"{RECORDER_NAME}.expect({s.receiver}, {s.method!r}, {key!r})",
                s.source,
                f"{RECORDER_NAME}.confirm({key!r})",
            ]
        )
    for position, lines in guarded.items():
        out["cells"][position]["source"] = "\n".join(lines)
    setup = "\n".join(
        [
            "import importlib.util as _lizyml_index_util",
            "_lizyml_index_spec = _lizyml_index_util.spec_from_file_location(",
            f"    '_lizyml_index_recorder_module', {str(RECORDER_FILE)!r}",
            ")",
            "_lizyml_index_module = _lizyml_index_util.module_from_spec(",
            "    _lizyml_index_spec",
            ")",
            "_lizyml_index_spec.loader.exec_module(_lizyml_index_module)",
            "from lizyml import Model as _lizyml_index_model",
            f"{RECORDER_NAME} = _lizyml_index_module.CallRecorder(_lizyml_index_model)",
            f"{RECORDER_NAME}.install()",
        ]
    )
    finish = f"print({RECORDER_NAME}.finish({keys!r}))"
    out["cells"] = [
        _code_cell(setup, out, "lizyml-index-recorder"),
        *out["cells"],
        _code_cell(finish, out, "lizyml-index-finish"),
    ]
    return out


# --- the retry policy ----------------------------------------------------------------


@dataclass
class Attempt:
    kernel: object
    workdir: Path
    error: str | None = None


def fresh_workdir(base: Path) -> Path:
    """A new directory holding a copy of ``notebooks/``."""
    base.mkdir(parents=True, exist_ok=True)
    return Path(
        shutil.copytree(NOTEBOOKS, Path(tempfile.mkdtemp(dir=base)) / "notebooks")
    )


def run_with_retries(
    execute: Callable[[Any, Path], None],
    *,
    new_kernel: Callable[[], Any],
    new_workdir: Callable[[], Path],
    log: list[Attempt],
    markers: Sequence[str] = NETWORK_ERROR_MARKERS,
    attempts: int = ATTEMPTS,
) -> None:
    """Run ``execute`` until it succeeds, retrying only network failures.

    Each attempt is appended to ``log`` with its own kernel and working
    directory. A ``CellExecutionError`` whose text contains a marker is retried
    while attempts remain; it is re-raised after the last attempt. Any other
    exception is re-raised at once.
    """
    for number in range(attempts):
        attempt = Attempt(new_kernel(), new_workdir())
        log.append(attempt)
        try:
            execute(attempt.kernel, attempt.workdir)
        except CellExecutionError as exc:
            attempt.error = str(exc)
            if number + 1 < attempts and any(m in attempt.error for m in markers):
                continue
            raise
        return


# --- execution -----------------------------------------------------------------------


class NativeKernelSpecs(KernelSpecManager):  # type: ignore[misc]
    """Always ipykernel's native kernel, run by this process's interpreter."""

    def get_kernel_spec(self, kernel_name: str) -> KernelSpec:
        spec = KernelSpec(resource_dir="", **get_kernel_dict())
        assert spec.argv[0] == sys.executable
        return spec


def native_kernel() -> AsyncKernelManager:
    return AsyncKernelManager(
        kernel_name="python3", kernel_spec_manager=NativeKernelSpecs()
    )


def _execute(
    nb: dict[str, Any], timeout: int, executed: list[Any]
) -> Callable[[Any, Path], None]:
    def run(kernel: Any, workdir: Path) -> None:
        node = nbformat.reads(json.dumps(nb), as_version=4)  # joins line lists
        client = NotebookClient(
            node,
            km=kernel,
            timeout=timeout,
            resources={"metadata": {"path": str(workdir)}},
        )
        client.execute(cleanup_kc=True)
        executed.append(node)

    return run


def run_instrumented(
    nb: dict[str, Any], *, workdir_base: Path, timeout: int = CELL_TIMEOUT
) -> list[Attempt]:
    """Instrument ``nb``, execute it with retries and check the sentinel."""
    instrumented = instrument_notebook(nb)
    executed: list[Any] = []
    log: list[Attempt] = []
    run_with_retries(
        _execute(instrumented, timeout, executed),
        new_kernel=native_kernel,
        new_workdir=lambda: fresh_workdir(workdir_base),
        log=log,
    )
    (node,) = executed
    last = node.cells[-1]
    text = "".join(
        o.get("text", "") for o in last.outputs if o.get("output_type") == "stream"
    )
    ix = load_examples_index()
    expected = len(ix.tagged_statements(nb, ix.check_notebook(nb)))
    assert text == f"{SENTINEL} {expected}\n", f"the recording cell printed {text!r}"
    return log
