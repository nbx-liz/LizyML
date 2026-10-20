"""Refuse every network call a tutorial notebook makes (H-0137, #385).

The notebooks read their datasets from ``notebooks/data/``. Both notebook
layers (the index execution and the slow execution test) load this file into
the kernel through a first cell and check it through a last cell:

- ``install()`` replaces the ``socket`` entry points a Python library reaches
  the network through (connect, name resolution, ``sendto``) with ones that
  record the call and raise ``NetworkRefused``. No address family or host is
  exempt: running every tutorial with these calls only logged found none
  (``docs/audits/2026-10-h0137/``).
- ``check()`` fails when any call was recorded, so a refusal that a library
  caught and passed over (or that a ``raises-exception`` cell allowed) still
  fails the run.

The guard binds the kernel process only. A subprocess the notebook starts
(``tutorial_codegen_export`` runs the generated ``train.py`` and ``predict.py``)
and a C extension that opens its own sockets are outside it.

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

# (owner, attribute) of every entry point the guard replaces.
_SOCKET_METHODS = ("connect", "connect_ex", "sendto")
_MODULE_FUNCTIONS = (
    "create_connection",
    "getaddrinfo",
    "gethostbyname",
    "gethostbyname_ex",
)


_INHERITED = object()


class NetworkRefused(OSError):
    """Raised in place of a network call inside a guarded notebook."""


class NetworkGuard:
    def __init__(self) -> None:
        self.attempts: list[str] = []
        self._originals: list[tuple[Any, str, Any]] = []

    def _refuse(self, name: str) -> Callable[..., Any]:
        def refused(*args: Any, **kwargs: Any) -> Any:
            call = f"{name}{args[1:] if name in _SOCKET_METHODS else args}"
            self.attempts.append(call)
            raise NetworkRefused(f"{MESSAGE}: {call}")

        return refused

    def install(self) -> NetworkGuard:
        for name in _SOCKET_METHODS:
            self._replace(socket.socket, name)
        for name in _MODULE_FUNCTIONS:
            self._replace(socket, name)
        return self

    def _replace(self, owner: Any, name: str) -> None:
        # socket.socket inherits connect & co. from _socket.socket, so the
        # attribute may not be the owner's own; uninstall then deletes it.
        self._originals.append((owner, name, owner.__dict__.get(name, _INHERITED)))
        setattr(owner, name, self._refuse(name))

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
    """The last cell: fail if any network call was attempted."""
    return f"{GUARD_NAME}.check()"
