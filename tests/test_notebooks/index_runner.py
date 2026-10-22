"""Execute a notebook and record that its index examples ran (H-0119 6.(b)).

The notebook is rewritten in memory only:

- a first cell loads ``index_recorder.py`` and wraps ``Model``;
- each ``index-example`` statement is preceded by ``expect(R, "m", key)``
  (``R`` is a ``Model``) and followed by ``confirm(key)`` (that call ran,
  outermost);
- a last cell checks that every tagged statement was confirmed and prints a
  sentinel, which the runner requires in the executed notebook.

A cell before the recorder installs ``network_guard.py`` and the last cell
checks it first, so a notebook that tries to use the network fails (H-0137).

Execution uses ipykernel's native kernel started with this process's
``sys.executable``, once, with a fresh copy of ``notebooks/`` (``data/``
included) as its working directory. H-0137 removed H-0119's retry of network
failures: the notebooks read vendored files.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import nbformat
from ipykernel.kernelspec import get_kernel_dict
from jupyter_client.kernelspec import KernelSpec, KernelSpecManager
from jupyter_client.manager import AsyncKernelManager
from nbclient import NotebookClient

from tests.test_notebooks import network_guard
from tests.test_notebooks.index_recorder import SENTINEL

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "notebooks"
RECORDER_FILE = Path(__file__).with_name("index_recorder.py")
RECORDER_NAME = "_lizyml_index_recorder"
CELL_TIMEOUT = 600

__all__ = ["SENTINEL", "instrument_notebook", "run_instrumented"]


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
    finish = "\n".join(
        [
            network_guard.check_source(),
            f"print({RECORDER_NAME}.finish({keys!r}))",
        ]
    )
    out["cells"] = [
        _code_cell(network_guard.setup_source(), out, "lizyml-network-guard"),
        _code_cell(setup, out, "lizyml-index-recorder"),
        *out["cells"],
        _code_cell(finish, out, "lizyml-index-finish"),
    ]
    return out


def fresh_workdir(base: Path) -> Path:
    """A new directory holding a copy of ``notebooks/``."""
    base.mkdir(parents=True, exist_ok=True)
    return Path(
        shutil.copytree(NOTEBOOKS, Path(tempfile.mkdtemp(dir=base)) / "notebooks")
    )


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


def run_instrumented(
    nb: dict[str, Any], *, workdir_base: Path, timeout: int = CELL_TIMEOUT
) -> Path:
    """Instrument ``nb``, execute it once and check the sentinel.

    Returns the working directory the notebook ran in.
    """
    workdir = fresh_workdir(workdir_base)
    node = nbformat.reads(json.dumps(instrument_notebook(nb)), as_version=4)
    client = NotebookClient(
        node,
        km=native_kernel(),
        timeout=timeout,
        resources={"metadata": {"path": str(workdir)}},
    )
    client.execute(cleanup_kc=True)
    last = node.cells[-1]
    text = "".join(
        o.get("text", "") for o in last.outputs if o.get("output_type") == "stream"
    )
    ix = load_examples_index()
    expected = len(ix.tagged_statements(nb, ix.check_notebook(nb)))
    assert text == f"{SENTINEL} {expected}\n", f"the recording cell printed {text!r}"
    return workdir
