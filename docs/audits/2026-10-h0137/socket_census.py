"""Run every tutorial with a *logging* cell and print what touched the network.

    uv run python docs/audits/2026-10-h0137/socket_census.py docs/audits/2026-10-h0137/socket_census.json

Measurement for H-0137: does anything in the kernel reach a Python-level
socket entry point (the fifteen that tests/test_notebooks/network_guard.py
refuses), and which files under data/ does each notebook open? The cell only
records; it refuses nothing. (ZMQ, the kernel's own channel, is C-level and is
not expected to show.) Each notebook runs in a fresh copy of notebooks/.
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
import builtins as _b, io as _io, os as _os, socket as _s
_SOCKLOG = []
_READS = set()
_DATA = _os.path.abspath("data")
import sys as _sys
_AUDIT = set()  # every socket.* audit event, _socket-level calls included
_sys.addaudithook(lambda e, a: _AUDIT.add(e) if e.startswith("socket.") else None)
def _wrap(owner, name):
    orig = getattr(owner, name)
    def f(*a, **k):
        _SOCKLOG.append((name, repr(a[:3])[:200]))
        return orig(*a, **k)
    setattr(owner, name, f)
for _n in ("bind", "connect", "connect_ex", "sendto", "sendmsg"):
    _wrap(_s.socket, _n)
for _n in ("create_connection", "getaddrinfo", "getfqdn", "gethostbyname", "gethostbyname_ex",
           "gethostbyaddr", "getnameinfo", "getprotobyname", "getservbyname", "getservbyport"):
    _wrap(_s, _n)
_open = _b.open
def _recording_open(file, *a, **k):
    handle = _open(file, *a, **k)
    mode = a[0] if a else k.get("mode", "r")
    if isinstance(file, (str, _os.PathLike)) and not set("wax+") & set(mode):
        p = _os.path.abspath(_os.fspath(file))
        if _os.path.dirname(p) == _DATA:
            _READS.add(_os.path.basename(p))
    return handle
_b.open = _io.open = _recording_open
'''
END_CELL = (
    "import json as _j; "
    "print('SOCKLOG ' + _j.dumps(_SOCKLOG)); print('READS ' + _j.dumps(sorted(_READS))); "
    "print('AUDIT ' + _j.dumps(sorted(_AUDIT)))"
)

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
