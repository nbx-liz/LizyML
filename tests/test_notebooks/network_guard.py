"""Hold a tutorial notebook to its offline contract (H-0137, #385).

The notebooks read their datasets from ``notebooks/data/``, as recorded in
``notebooks/data/manifest.toml``, and use no network. Both notebook layers (the
index execution and the slow execution test) load this file into the kernel
through a first cell and check it through a last cell:

- ``install()`` replaces the ``socket`` entry points a Python library reaches
  the network through with ones that record the call and raise
  ``NetworkRefused``. Which ones is decided by a table that classifies every
  public function of ``socket`` and every public method of ``socket.socket``
  (``REFUSED_*`` and ``NOT_REFUSED_*`` below, each kept with its reason); a
  test fails on a public name the table does not classify, so the list stays
  closed as Python changes. Refused are the methods that take a destination
  address and the functions that resolve a name or address or look one up in a
  system name database. No address family or host is exempt, ``AF_UNIX`` and
  loopback included: running every tutorial with these calls only logged found
  none (``docs/audits/2026-10-h0137/``).
- It also records every file in the ``data/`` directory under its working
  directory that the kernel opens for reading through ``open`` / ``io.open``:
  the open must succeed and its mode must hold none of ``w``, ``a``, ``x``,
  ``+``.
- ``check(expected)`` fails when any network call was recorded (so a refusal a
  library caught and passed over, or a ``raises-exception`` cell allowed, still
  fails the run), or when the ``data/`` files read are not exactly
  ``expected``, the files the manifest records for this notebook.

The guard binds the kernel process and Python-level calls only. A subprocess
the notebook starts (``tutorial_codegen_export`` runs the generated
``predict.py`` and ``test_equivalence.py`` on local files), code that calls the
``_socket`` extension module directly, and a C extension that opens its own
sockets or files are outside it.

This module runs inside the kernel, so its kernel-side part imports only the
standard library.
"""

from __future__ import annotations

import builtins
import io
import os
import socket
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

GUARD_FILE = Path(__file__)
GUARD_NAME = "_lizyml_network_guard"
MESSAGE = "H-0137: a tutorial notebook tried to use the network"
READS_MESSAGE = "H-0137: the notebook's data/ reads differ from the manifest"
MANIFEST = GUARD_FILE.resolve().parents[2] / "notebooks" / "data" / "manifest.toml"

# Every public method of socket.socket (CPython 3.10-3.13 on Linux), classified.
REFUSED_METHODS = {
    "connect": "connects to a destination address",
    "connect_ex": "connects to a destination address",
    "sendto": "sends to a destination address",
    "sendmsg": "sends to a destination address when one is given",
}
_LOCAL_ENDPOINT = "creates or serves a local endpoint; reaches no other host by itself"
_CONNECTED = "uses a socket that is already connected (only through a refused call)"
_LOCAL = "reads or sets local socket state; no traffic"
NOT_REFUSED_METHODS = {
    "accept": _LOCAL_ENDPOINT,
    "bind": _LOCAL_ENDPOINT,
    "listen": _LOCAL_ENDPOINT,
    "recv": _CONNECTED,
    "recv_into": _CONNECTED,
    "recvfrom": _CONNECTED,
    "recvfrom_into": _CONNECTED,
    "recvmsg": _CONNECTED,
    "recvmsg_into": _CONNECTED,
    "send": _CONNECTED,
    "sendall": _CONNECTED,
    "sendfile": _CONNECTED,
    "sendmsg_afalg": "AF_ALG, the kernel crypto interface; no address",
    "shutdown": _CONNECTED,
    "close": _LOCAL,
    "detach": _LOCAL,
    "dup": _LOCAL,
    "fileno": _LOCAL,
    "get_inheritable": _LOCAL,
    "getblocking": _LOCAL,
    "getpeername": _LOCAL,
    "getsockname": _LOCAL,
    "getsockopt": _LOCAL,
    "gettimeout": _LOCAL,
    "makefile": _LOCAL,
    "set_inheritable": _LOCAL,
    "setblocking": _LOCAL,
    "setsockopt": _LOCAL,
    "settimeout": _LOCAL,
}
# Every public function of socket (the same interpreters), classified.
_RESOLVES = "resolves a host name or address (DNS or another name service)"
_NAME_DB = "looks up a system name database, which a name service may serve remotely"
REFUSED_FUNCTIONS = {
    "create_connection": "resolves and connects to a destination",
    "getaddrinfo": _RESOLVES,
    "getfqdn": _RESOLVES,
    "gethostbyaddr": _RESOLVES,
    "gethostbyname": _RESOLVES,
    "gethostbyname_ex": _RESOLVES,
    "getnameinfo": _RESOLVES,
    "getprotobyname": _NAME_DB,
    "getservbyname": _NAME_DB,
    "getservbyport": _NAME_DB,
}
_CONVERTS = "converts a value; no traffic"
NOT_REFUSED_FUNCTIONS = {
    "create_server": _LOCAL_ENDPOINT,
    "socketpair": "a connected pair inside this process",
    "fromfd": _LOCAL,
    "dup": _LOCAL,
    "close": _LOCAL,
    "getdefaulttimeout": _LOCAL,
    "setdefaulttimeout": _LOCAL,
    "gethostname": "this machine's own name (uname); no lookup",
    "sethostname": "sets this machine's own name; no lookup",
    "has_dualstack_ipv6": "probes local IPv6 support; no traffic",
    "if_indextoname": "local interface table",
    "if_nameindex": "local interface table",
    "if_nametoindex": "local interface table",
    "send_fds": "sends through socket.sendmsg, which is refused",
    "recv_fds": _CONNECTED,
    "CMSG_LEN": _CONVERTS,
    "CMSG_SPACE": _CONVERTS,
    "htonl": _CONVERTS,
    "htons": _CONVERTS,
    "ntohl": _CONVERTS,
    "ntohs": _CONVERTS,
    "inet_aton": _CONVERTS,
    "inet_ntoa": _CONVERTS,
    "inet_ntop": _CONVERTS,
    "inet_pton": _CONVERTS,
}
SOCKET_METHODS = tuple(REFUSED_METHODS)
MODULE_FUNCTIONS = tuple(REFUSED_FUNCTIONS)
_WRITE_MODES = frozenset("wax+")

_INHERITED = object()


class NetworkRefused(OSError):
    """Raised in place of a network call inside a guarded notebook."""


class NetworkGuard:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.attempts: list[str] = []
        self.reads: set[str] = set()
        self._originals: list[tuple[Any, str, Any]] = []

    def _refuse(self, name: str) -> Callable[..., Any]:
        def refused(*args: Any, **kwargs: Any) -> Any:
            call = f"{name}{args[1:] if name in SOCKET_METHODS else args}"
            self.attempts.append(call)
            raise NetworkRefused(f"{MESSAGE}: {call}")

        return refused

    def _recording_open(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def recording_open(file: Any, *args: Any, **kwargs: Any) -> Any:
            handle = original(file, *args, **kwargs)  # a failed open records nothing
            mode = args[0] if args else kwargs.get("mode", "r")
            if isinstance(file, (str, os.PathLike)) and not _WRITE_MODES & set(mode):
                path = Path(os.path.abspath(os.fspath(file)))
                if path.parent == self.data_dir:
                    self.reads.add(path.name)
            return handle

        return recording_open

    def install(self) -> NetworkGuard:
        for name in SOCKET_METHODS:
            self._replace(socket.socket, name, self._refuse(name))
        for name in MODULE_FUNCTIONS:
            self._replace(socket, name, self._refuse(name))
        recording_open = self._recording_open(builtins.open)
        self._replace(builtins, "open", recording_open)
        self._replace(io, "open", recording_open)
        return self

    def _replace(self, owner: Any, name: str, replacement: Any) -> None:
        # socket.socket inherits connect & co. from _socket.socket, so the
        # attribute may not be the owner's own; uninstall then deletes it.
        self._originals.append((owner, name, owner.__dict__.get(name, _INHERITED)))
        setattr(owner, name, replacement)

    def uninstall(self) -> None:
        while self._originals:
            owner, name, original = self._originals.pop()
            if original is _INHERITED:
                delattr(owner, name)
            else:
                setattr(owner, name, original)

    def check(self, expected: Sequence[str] = ()) -> None:
        if self.attempts:
            raise AssertionError(f"{MESSAGE}: {self.attempts}")
        if self.reads != set(expected):
            raise AssertionError(
                f"{READS_MESSAGE}: read {sorted(self.reads)}, "
                f"the manifest records {sorted(expected)}"
            )


def install(data_dir: Path | None = None) -> NetworkGuard:
    """Install a guard; ``data_dir`` defaults to ``data/`` under the cwd."""
    return NetworkGuard(data_dir or Path(os.getcwd()).resolve() / "data").install()


def setup_source() -> str:
    """The first cell: load this file into the kernel and install the guard."""
    return "\n".join(
        [
            "import importlib.util as _lizyml_guard_util",
            "_lizyml_guard_spec = _lizyml_guard_util.spec_from_file_location(",
            f"    '_lizyml_network_guard_module', {str(GUARD_FILE)!r}",
            ")",
            "_lizyml_guard_module = _lizyml_guard_util.module_from_spec(",
            "    _lizyml_guard_spec",
            ")",
            "_lizyml_guard_spec.loader.exec_module(_lizyml_guard_module)",
            f"{GUARD_NAME} = _lizyml_guard_module.install()",
        ]
    )


def check_source(expected: Sequence[str] = ()) -> str:
    """The last cell: no network call, and exactly ``expected`` read from data/."""
    return f"{GUARD_NAME}.check({sorted(expected)!r})"


def manifest_files(notebook: str, manifest: Path = MANIFEST) -> list[str]:
    """The ``data/`` files the manifest records for ``notebook`` (test side)."""
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover - Python 3.10; pytest itself depends on tomli there
        import tomli as tomllib
    entries = tomllib.loads(manifest.read_text(encoding="utf-8"))["dataset"]
    return sorted(e["file"] for e in entries if notebook in e["notebooks"])
