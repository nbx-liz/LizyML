"""Record which ``Model`` methods a notebook really runs (H-0119 6.(b)).

``index_runner.py`` loads this file by path into the notebook's kernel, wraps
every public callable of ``Model`` (inherited mixin methods included) and
records the outermost calls only: a public method called from inside another
``Model`` method is not recorded. Around each ``index-example`` statement the
instrumented notebook calls :meth:`CallRecorder.expect` and
:meth:`CallRecorder.confirm`, and its last cell calls
:meth:`CallRecorder.finish`.

Standard library only: the module runs in the kernel and in the unit tests.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Iterable
from typing import Any

SENTINEL = "LIZYML-INDEX-RECORDED"


class CallRecorder:
    """Wraps a class's public callables and records outermost calls."""

    def __init__(self, cls: type) -> None:
        self._cls = cls
        self.calls: list[tuple[str, int | None]] = []
        self._depth = 0
        self._saved: dict[str, tuple[bool, Any]] | None = None
        self._pending: tuple[str, str, int, int] | None = None
        self._confirmed: set[str] = set()

    # --- wrapping ---------------------------------------------------------

    def _wrap(
        self, name: str, func: Callable[..., Any], *, bound: bool
    ) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if self._depth == 0:
                self.calls.append((name, id(args[0]) if bound and args else None))
            self._depth += 1
            try:
                return func(*args, **kwargs)
            finally:
                self._depth -= 1

        return wrapper

    def install(self) -> None:
        if self._saved is not None:
            raise RuntimeError("the recorder is already installed")
        saved: dict[str, tuple[bool, Any]] = {}
        replacements: dict[str, Any] = {}
        for name, value in inspect.getmembers(self._cls):
            if name.startswith("_") or not callable(value):
                continue  # properties and data are not methods
            raw = inspect.getattr_static(self._cls, name)
            if isinstance(raw, staticmethod):
                new: Any = staticmethod(self._wrap(name, raw.__func__, bound=False))
            elif isinstance(raw, classmethod):
                new = classmethod(self._wrap(name, raw.__func__, bound=True))
            elif inspect.isfunction(raw):
                new = self._wrap(name, raw, bound=True)
            else:
                raise TypeError(
                    f"{self._cls.__name__}.{name}: cannot wrap {type(raw).__name__}"
                )
            saved[name] = (name in vars(self._cls), vars(self._cls).get(name))
            replacements[name] = new
        for name, new in replacements.items():
            setattr(self._cls, name, new)
        self._saved = saved

    def uninstall(self) -> None:
        if self._saved is None:
            raise RuntimeError("the recorder is not installed")
        for name, (own, original) in self._saved.items():
            if own:
                setattr(self._cls, name, original)
            else:
                delattr(self._cls, name)
        self._saved = None

    # --- the checks the instrumented notebook calls --------------------------

    def expect(self, receiver: object, method: str, key: str) -> None:
        """Before statement ``key``: its receiver is a ``Model``."""
        assert self._pending is None, (
            f"statement {self._pending[0]} was never confirmed"
        )
        assert isinstance(receiver, self._cls), (
            f"statement {key}: the receiver of {method}() is a "
            f"{type(receiver).__name__}, not a {self._cls.__name__}"
        )
        self._pending = (key, method, id(receiver), len(self.calls))

    def confirm(self, key: str) -> None:
        """After statement ``key``: it ran exactly its declared call, outermost."""
        assert self._pending is not None and self._pending[0] == key, (
            f"statement {key}: confirmed without a matching expect"
        )
        _, method, receiver, start = self._pending
        made = self.calls[start:]
        assert made == [(method, receiver)], (
            f"statement {key}: expected one outermost {method}() on the declared "
            f"receiver, recorded {made}"
        )
        self._pending = None
        self._confirmed.add(key)

    def finish(self, keys: Iterable[str]) -> str:
        """Every tagged statement was confirmed; returns the sentinel line."""
        keys = list(keys)
        assert self._pending is None, (
            f"statement {self._pending[0]} was never confirmed"
        )
        missing = [k for k in keys if k not in self._confirmed]
        assert not missing, f"tagged statements that never ran: {missing}"
        return f"{SENTINEL} {len(keys)}"
