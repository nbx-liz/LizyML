"""Fail a tutorial notebook that loads data over the network (H-0137, #385).

The tutorials read their datasets from ``notebooks/data/``. The regression this
guards against is a maintainer bringing a remote load back, which would put
the release gate back at the mercy of a third-party host: any load that goes
through the two entry points below, as ``fetch_openml``, ``pandas.read_csv`` of
a URL and ``urllib`` do (``docs/audits/2026-10-h0137/``).

Both notebook layers (the index execution and the slow execution test) load
this file into the kernel through a first cell and check it through a last
cell:

- ``install()`` replaces ``socket.getaddrinfo`` and ``socket.socket.connect``,
  the two Python entry points an HTTP client goes through to resolve a host
  and to open a connection, with ones that record the call and raise
  ``NetworkRefused``.
- ``check()`` fails when any call was recorded, so a refusal that a library
  caught and passed over (or that a ``raises-exception`` cell allowed) still
  fails the run.

The claim is exactly that: these two entry points are refused. It is not a
sandbox. Other socket calls (``connect_ex``, ``sendto``, ``bind``, ...), direct
``_socket`` use, C extensions and subprocesses are outside it; a stronger
guarantee would need OS-level isolation and its own Proposal.

This module runs inside the kernel, so it imports only the standard library.
"""

from __future__ import annotations

import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

GUARD_FILE = Path(__file__)
GUARD_NAME = "_lizyml_network_guard"
MESSAGE = "H-0137: a tutorial notebook tried to use the network"

# (owner, attribute) of the two refused entry points.
REFUSED = ((socket, "getaddrinfo"), (socket.socket, "connect"))

_INHERITED = object()


class NetworkRefused(OSError):
    """Raised in place of a network call inside a guarded notebook."""


class NetworkGuard:
    def __init__(self) -> None:
        self.attempts: list[str] = []
        self._originals: list[tuple[Any, str, Any]] = []

    def _refuse(self, name: str) -> Callable[..., Any]:
        def refused(*args: Any, **kwargs: Any) -> Any:
            shown = args[1:] if name == "connect" else args  # drop the socket
            call = f"{name}{shown}"
            self.attempts.append(call)
            raise NetworkRefused(f"{MESSAGE}: {call}")

        return refused

    def install(self) -> NetworkGuard:
        for owner, name in REFUSED:
            # socket.socket inherits connect from _socket.socket, so the
            # attribute may not be the owner's own; uninstall then deletes it.
            original = owner.__dict__.get(name, _INHERITED)
            self._originals.append((owner, name, original))
            setattr(owner, name, self._refuse(name))
        return self

    def uninstall(self) -> None:
        while self._originals:
            owner, name, original = self._originals.pop()
            if original is _INHERITED:
                delattr(owner, name)
            else:
                setattr(owner, name, original)

    def check(self) -> None:
        if self.attempts:
            raise AssertionError(f"{MESSAGE}: {self.attempts}")


def install() -> NetworkGuard:
    return NetworkGuard().install()


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


def check_source() -> str:
    """The last cell: fail if either entry point was called."""
    return f"{GUARD_NAME}.check()"
