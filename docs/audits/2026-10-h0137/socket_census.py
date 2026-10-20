"""Run every tutorial with a socket *logging* cell and print what touched the network.

    uv run python docs/audits/2026-10-h0137/socket_census.py docs/audits/2026-10-h0137/socket_census.json

Measurement for H-0137: does anything in the kernel open a Python-level socket
connection or resolve a name? (ZMQ is C-level and is not expected to show.)
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[3]
LOG_CELL = r'''
import socket as _s
_SOCKLOG = []
def _wrap(owner, name):
    orig = getattr(owner, name)
    def f(*a, **k):
        _SOCKLOG.append((name, repr(a[:3])[:200]))
        return orig(*a, **k)
    setattr(owner, name, f)
for _n in ("connect", "connect_ex", "sendto"):
    _wrap(_s.socket, _n)
for _n in ("getaddrinfo", "create_connection", "gethostbyname", "gethostbyname_ex"):
    _wrap(_s, _n)
'''
END_CELL = "import json as _j; print('SOCKLOG ' + _j.dumps(_SOCKLOG))"

results = {}
for path in sorted((REPO / "notebooks").glob("*.ipynb")):
    nb = nbformat.read(str(path), as_version=4)
    nb.cells.insert(0, nbformat.v4.new_code_cell(LOG_CELL))
    nb.cells.append(nbformat.v4.new_code_cell(END_CELL))
    work = Path(tempfile.mkdtemp()) / "notebooks"
    shutil.copytree(REPO / "notebooks", work)
    client = NotebookClient(nb, timeout=900, kernel_name="python3",
                            resources={"metadata": {"path": str(work)}})
    try:
        client.execute()
    except Exception as exc:  # report and continue with the next notebook
        results[path.name] = f"FAILED: {type(exc).__name__}: {str(exc)[-600:]}"
        print(path.name, results[path.name], flush=True)
        continue
    text = "".join(o.get("text", "") for o in nb.cells[-1].outputs)
    results[path.name] = text.strip()
    print(path.name, text.strip()[:2000], flush=True)

Path(sys.argv[1]).write_text(json.dumps(results, indent=1))
