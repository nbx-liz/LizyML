"""Documents describing the current system must not state a stale version constant.

Three sites in ``ARCHITECTURE.md`` stated ``format_version`` 1 while
``persistence/exporter.py`` has defined 2 since H-0070. They drifted together and
CI stayed green, because nothing compared a constant a document *states* against
the value the code *defines*.

This closes the class rather than the three instances: the population is every
version constant stated by a document that describes the system as it is now,
and the expected value is read from the code at import time, so a future bump
cannot leave such a document behind without failing here.

Two guards keep the check honest:

* **Append-only records are excluded, by name and with a reason.** ``HISTORY.md``
  and ``CHANGELOG.md`` record decisions and releases as they happened, and
  ``PLAN.md`` is a roadmap; each correctly states ``format_version=1`` where it
  narrates the past or the load-compatibility path. Forcing those to the current
  value would falsify the record. Every excluded path is asserted to exist, so a
  rename fails loudly instead of silently widening the population.
* **The scan must find sites.** If the documents are restructured so the grammar
  below stops matching, the guard test fails rather than the suite reporting a
  vacuous pass (DC1).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lizyml.config.loader import SUPPORTED_CONFIG_VERSIONS
from lizyml.persistence.exporter import FORMAT_VERSION

REPO = Path(__file__).resolve().parents[2]

#: Documents that describe the system as it currently is. A version constant
#: stated here is a claim about today, so it must match the code.
CURRENT_DOCS: tuple[str, ...] = ("ARCHITECTURE.md", "BLUEPRINT.md", "README.md")

#: Documents deliberately outside the population, and why. These state past
#: version values on purpose.
EXCLUDED: dict[str, str] = {
    "HISTORY.md": "append-only decision record; states the value each decision fixed",
    "CHANGELOG.md": "append-only release record; narrates the 1 -> 2 bump",
    "PLAN.md": "roadmap; refers to format_version=1 as the load-compatibility case",
}

#: Directory trees outside the population, and why. Audit archives quote the
#: stale values they found and the rejection inputs they exercised; forcing them
#: current would destroy the finding they record.
EXCLUDED_DIRS: dict[str, str] = {
    "docs/audits": "archived audit records; quote the values they were reporting on",
}

#: What counts as *stating* the constant: its name, a separator, and the value
#: token that follows. Both spellings the repository uses are covered —
#: ``format_version=1`` in diagram labels and ``FORMAT_VERSION = 1`` in class
#: blocks. A bare mention of the field name, or a rejection example such as
#: "unknown version (99, 0)", carries no separator, so it is not a declaration
#: and is not in scope.
#:
#: Only the *prefix* is matched here — name and separator. Everything after it on
#: the line is the remainder, classified below. Matching a token instead of a
#: remainder is what leaves a boundary open: any token pattern stops somewhere,
#: and whatever follows the stop is then invisible.
DECLARATION: dict[str, re.Pattern[str]] = {
    name: re.compile(rf"\b{name}\s*[=:]\s*", re.IGNORECASE)
    for name in ("format_version", "config_version")
}

#: The characters that legitimately end a stated value in these documents:
#: end of line, whitespace, or one of the closing marks the surrounding syntax
#: uses — mermaid's quote and bracket, Markdown's backtick, and Japanese
#: parentheses and full stop. A value is accepted only when the character right
#: after its digits is one of these, so ``2bogus``, ``2-bogus`` and ``2.5`` are
#: all rejected rather than read as ``2`` (DC2).
VALUE = re.compile(r"(\d+)(?=$|[\s\"'`)\]}>,;）」』。])")

#: The one other thing this position legitimately holds is a type annotation, and
#: it must be the *whole* remainder. ``+config_version: int`` states the field's
#: type; ``config_version: int = 999`` states a type *and* an initialiser, so it
#: is not an annotation and must not be read as one — it is rejected, because
#: which of the two the document meant is not this check's guess to make.
ANNOTATION = re.compile(r"([A-Za-z_][\w.\[\]]*)\s*$")

#: The one other thing this position legitimately holds. ``ARCHITECTURE.md``'s
#: mermaid class blocks declare the *field* as ``+config_version: int``, which
#: states the constant's type and not its value. Every admitted spelling is
#: named here and asserted to be in use below, so a new one fails as unreadable
#: instead of quietly widening the grammar one tolerated form at a time.
TYPE_ANNOTATIONS: frozenset[str] = frozenset({"int"})

#: Values the code defines. A current-system document may state only these.
ALLOWED: dict[str, set[int]] = {
    "format_version": {FORMAT_VERSION},
    "config_version": set(SUPPORTED_CONFIG_VERSIONS),
}

#: Every document that must contribute at least one site. A count alone is not
#: enough: with a bare minimum, ARCHITECTURE.md could drop out of the scan
#: entirely and the other documents would still satisfy it — which is exactly
#: the drift this check exists for (DC1).
MUST_CONTRIBUTE: tuple[str, ...] = ("ARCHITECTURE.md", "BLUEPRINT.md")

#: Sites present when this check was written. Fewer means the scan stopped
#: looking; this is the coarse guard, MUST_CONTRIBUTE is the sharp one.
MIN_SITES = 7


def _is_excluded_dir(path: Path) -> bool:
    rel = path.relative_to(REPO).as_posix()
    return any(rel.startswith(f"{d}/") for d in EXCLUDED_DIRS)


def _docs() -> list[Path]:
    return [
        p
        for p in (
            [REPO / name for name in CURRENT_DOCS] + sorted(REPO.glob("docs/**/*.md"))
        )
        if p.is_file() and not _is_excluded_dir(p)
    ]


def _scan() -> tuple[
    list[tuple[str, str, int, int]],
    list[tuple[str, str, int, str]],
    list[tuple[str, str, int, str]],
]:
    """Classify every declaration; the three outcomes are exhaustive.

    Returns ``(sites, annotations, rejected)``. ``sites`` carries
    ``(constant, relative path, line number, stated value)`` for a well-formed
    value; ``annotations`` and ``rejected`` carry the raw token instead. Nothing
    is dropped: a token that is neither a value nor a named type annotation
    lands in ``rejected`` and fails the suite.
    """
    sites: list[tuple[str, str, int, int]] = []
    annotations: list[tuple[str, str, int, str]] = []
    rejected: list[tuple[str, str, int, str]] = []
    for doc in _docs():
        rel = doc.relative_to(REPO).as_posix()
        for lineno, line in enumerate(
            doc.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for name, pattern in DECLARATION.items():
                for m in pattern.finditer(line):
                    # The remainder, not a token: a second declaration later on
                    # the same line gets its own match and its own remainder,
                    # rather than being swallowed by the first one's token.
                    rest = line[m.end() :]
                    value = VALUE.match(rest)
                    annotation = ANNOTATION.fullmatch(rest)
                    if value:
                        sites.append((name, rel, lineno, int(value.group(1))))
                    elif annotation and annotation.group(1) in TYPE_ANNOTATIONS:
                        annotations.append((name, rel, lineno, annotation.group(1)))
                    else:
                        rejected.append((name, rel, lineno, rest))
    return sites, annotations, rejected


SITES, ANNOTATIONS, REJECTED = _scan()


def test_current_documents_exist() -> None:
    """Every document named in the population must actually be there."""
    missing = [name for name in CURRENT_DOCS if not (REPO / name).is_file()]
    assert not missing, (
        f"named current-system documents are missing: {missing}. Update "
        "CURRENT_DOCS — a renamed document must not drop out of the check "
        "silently."
    )


def test_excluded_documents_exist() -> None:
    """Exclusions must name real paths, so a rename cannot widen the scope quietly."""
    missing = [name for name in EXCLUDED if not (REPO / name).is_file()]
    missing += [name for name in EXCLUDED_DIRS if not (REPO / name).is_dir()]
    assert not missing, (
        f"excluded paths are missing: {missing}. Update EXCLUDED / EXCLUDED_DIRS "
        "with the new name and its reason, or drop the entry deliberately."
    )


def test_every_named_document_contributes_a_site() -> None:
    """Each central document must be reached, not merely the population as a whole."""
    seen = {path for _, path, _, _ in SITES}
    silent = [name for name in MUST_CONTRIBUTE if name not in seen]
    assert not silent, (
        f"these documents contributed no declared-version site: {silent}. They "
        "are where this check's drift was found, so a scan that no longer "
        "reaches them is a broken check, not a clean result."
    )


def test_named_type_annotations_are_used() -> None:
    """Every spelling admitted as a type annotation must actually occur.

    An unused entry is an unreviewed hole in the grammar: it admits a token
    nothing in the documents produces, so it can only ever let something
    unexpected through.
    """
    seen = {token for _, _, _, token in ANNOTATIONS}
    unused = sorted(TYPE_ANNOTATIONS - seen)
    assert not unused, (
        f"these TYPE_ANNOTATIONS entries match nothing in the documents: "
        f"{unused}. Remove them, or update them to the spelling now in use."
    )


def test_no_declaration_is_unreadable() -> None:
    """A declaration whose value cannot be read is a failure, never a skip.

    This is the half of the grammar that closes it. Accepted: the constant's
    name, a separator, and a bare integer, or one of the named type annotations.
    Rejected — and reported here rather than dropped: anything else the same
    position can hold, such as ``format_version=2bogus``, ``format_version=2.5``,
    or a separator with no value after it. Dropping those would make the check
    report "clean" for a document it could not actually read (DC1), and a value
    pattern with no trailing boundary would read ``2bogus`` as ``2`` (DC2).
    """
    assert not REJECTED, (
        f"these documents state a version constant this check cannot read: "
        f"{REJECTED}. Each tuple is (constant, path, line, the raw value token). "
        "Fix the document, or widen the accepted grammar deliberately — but do "
        "not let an unreadable declaration pass as if it were absent."
    )


def test_documents_are_scanned() -> None:
    """The scan must actually reach documents and find sites."""
    assert len(SITES) >= MIN_SITES, (
        f"expected at least {MIN_SITES} declared-version sites, found "
        f"{len(SITES)}: {SITES}. Either the documents were restructured or "
        "GRAMMAR no longer matches how they state these constants."
    )


@pytest.mark.parametrize(
    ("name", "path", "lineno", "stated"),
    SITES,
    ids=[f"{n}-{p}:{ln}" for n, p, ln, _ in SITES],
)
def test_declared_version_matches_code(
    name: str, path: str, lineno: int, stated: int
) -> None:
    """Every stated constant must be one the code currently defines."""
    allowed = ALLOWED[name]
    assert stated in allowed, (
        f"{path}:{lineno} states {name}={stated}, but the code defines "
        f"{sorted(allowed)}. Update the document, or the constant, so the two "
        "agree — documents outrank code in this repository's hierarchy, so a "
        "stale one is a specification defect, not a cosmetic one."
    )
