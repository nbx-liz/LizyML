"""Check that every declared dependency floor is the installed version (H-0126, #295).

The ``Quality (lowest-direct deps)`` CI lane installs LizyML with all extras at
``--resolution lowest-direct`` and runs this before the suite. It reads the
declaration from the installed package's metadata (``Requires-Dist``), so the
floors are never copied out of ``pyproject.toml``, and fails unless every
requirement -- core and every extra -- is installed at exactly its floor.

The accepted requirement grammar is closed: a name, exactly one ``>=``
specifier, and either no marker or exactly ``extra == "<name>"``. Anything else
(an upper bound, ``==``, two specifiers, a URL, another marker) is reported as
unreadable and fails the check rather than being skipped.

Usage::

    .venv/bin/python scripts/check_dependency_floors.py
"""

from __future__ import annotations

import importlib.metadata
import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version

DISTRIBUTION = "lizyml"

_EXTRA_MARKER = re.compile(r'extra == "([A-Za-z0-9][A-Za-z0-9._-]*)"')


@dataclass(frozen=True)
class Row:
    """One declared requirement and what is installed for it."""

    requirement: str
    name: str | None
    extra: str | None
    floor: str | None
    installed: str | None
    ok: bool
    reason: str


def _unreadable(text: str, reason: str) -> Row:
    return Row(text, None, None, None, None, False, f"unreadable: {reason}")


def check_requirement(text: str, installed_version: Callable[[str], str | None]) -> Row:
    """Judge one ``Requires-Dist`` entry against the installed version."""
    try:
        req = Requirement(text)
    except InvalidRequirement as exc:
        return _unreadable(text, str(exc))
    if req.url is not None:
        return _unreadable(text, "a URL requirement has no floor")
    if req.extras:
        return _unreadable(text, "a requirement with extras is not a plain floor")
    specs = list(req.specifier)
    if len(specs) != 1 or specs[0].operator != ">=":
        return _unreadable(text, "expected exactly one '>=' specifier")
    extra: str | None = None
    if req.marker is not None:
        match = _EXTRA_MARKER.fullmatch(str(req.marker))
        if match is None:
            return _unreadable(text, f"unsupported marker {str(req.marker)!r}")
        extra = match.group(1)
    floor = specs[0].version
    installed = installed_version(req.name)
    if installed is None:
        return Row(text, req.name, extra, floor, None, False, "not installed")
    try:
        equal = Version(installed) == Version(floor)
    except InvalidVersion as exc:
        return Row(text, req.name, extra, floor, installed, False, str(exc))
    reason = "at the floor" if equal else "installed version is not the floor"
    return Row(text, req.name, extra, floor, installed, equal, reason)


def check(
    requirements: Iterable[str], installed_version: Callable[[str], str | None]
) -> list[Row]:
    """Judge every requirement; an empty declaration is itself a failure."""
    rows = [check_requirement(text, installed_version) for text in requirements]
    if not rows:
        rows.append(_unreadable("<none>", f"{DISTRIBUTION} declares no requirement"))
    return rows


def installed_version(name: str) -> str | None:
    """The installed version of distribution *name*, or ``None``."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def render(rows: list[Row]) -> str:
    """The resolved-version table the lane prints."""
    header = ("package", "extra", "floor", "installed", "result")
    body = [
        (
            row.name or row.requirement,
            row.extra or "-",
            row.floor or "-",
            row.installed or "-",
            ("ok" if row.ok else "FAIL") + f" ({row.reason})",
        )
        for row in rows
    ]
    widths = [max(len(r[i]) for r in [header, *body]) for i in range(len(header))]
    lines = [
        "  ".join(cell.ljust(w) for cell, w in zip(r, widths, strict=True))
        for r in [header, *body]
    ]
    return "\n".join(line.rstrip() for line in lines)


def main() -> int:
    try:
        requirements = importlib.metadata.requires(DISTRIBUTION) or []
    except importlib.metadata.PackageNotFoundError:
        print(f"{DISTRIBUTION} is not installed", file=sys.stderr)
        return 1
    rows = check(requirements, installed_version)
    print(render(rows))
    failed = [row for row in rows if not row.ok]
    if failed:
        print(f"\n{len(failed)} of {len(rows)} requirement(s) are not at their floor.")
        return 1
    print(f"\nAll {len(rows)} requirement(s) are installed at their floor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
