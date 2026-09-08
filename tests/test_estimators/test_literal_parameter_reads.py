"""Reads of a caller-spelled parameter dict under one literal spelling.

The write direction has been scanned since H-0093: which places send a parameter
to the estimator. The **read** direction had no scan, and it cost a defect --
`_extract_feval_metadata` read `adapter.params.get("metric")` by the literal
name while `_build_params` read the same parameter by identity, so a custom
metric written as `metrics` trained correctly and was dropped from the export,
leaving generated code that would not run (H-0094 decision 9, review round 13).

That round's record answered the question in prose: "a grep returns four
candidates". The grep's pattern was nowhere written down, so the enumeration
could not be re-run -- which the rounds 13-14 monitor named as the one
population in this PR that got no executable declaration while every other one
did. This file is that declaration.

**What it does not claim.** It finds subscript and `.get`/`.pop` reads with a
string literal key on a name that looks like a parameter dict. It is a syntactic
scan, not a type analysis, so a parameter dict held under a name outside
`_DICT_NAMES` is invisible to it. The names are a module constant so that "the
scan missed it" stays checkable, and the hostile-source cases below are what
make the scan itself falsifiable rather than merely present.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

#: Identifiers that hold a dict whose keys the *caller* chose. A read of one of
#: these under a single literal spelling is the shape this file exists to find.
_DICT_NAMES = frozenset(
    {
        "params",
        "user_params",
        "model_params",
        "extra_params",
        "effective_params",
        # `isotonic.py` calls the caller dict `user` and the merged one
        # `merged`. Round 13 populated its enumeration by reading rather
        # than by running a scan, and a narrower name filter is precisely
        # how that enumeration missed things.
        "user",
        "merged",
    }
)

#: Reads that are correct under one spelling, each with the reason. A name is
#: only allowed here when the dict it reads has already been normalised, or when
#: the parameter has exactly one accepted spelling.
_ALLOWED: dict[tuple[str, str], str] = {
    (
        "lizyml/estimators/lgbm/adapter.py",
        "metric",
    ): "read after `_pop_by_identity` has popped every spelling and rewritten it",
    (
        "lizyml/estimators/lgbm/adapter.py",
        "objective",
    ): "read after `_pop_by_identity` has popped every spelling and rewritten it",
    (
        "lizyml/estimators/lgbm/adapter.py",
        "num_class",
    ): "LizyML sets it; not a name a caller can write",
    (
        "lizyml/estimators/lgbm/smart_params.py",
        "max_depth",
    ): "`max_depth` has no alias in LightGBM's registry; asserted below",
    (
        "lizyml/calibration/isotonic.py",
        "min_data_in_leaf",
    ): "written by the calibrator's own ratio; #280's class, recorded",
    (
        "lizyml/calibration/isotonic.py",
        "monotone_constraints",
    ): "forced under the canonical spelling, asserted in the calibration tests",
    (
        "lizyml/calibration/isotonic.py",
        "verbosity",
    ): "forced under the canonical spelling, asserted in the calibration tests",
    (
        "lizyml/calibration/isotonic.py",
        "verbose",
    ): "popped, not read: the alias is removed so the forced canonical wins",
    (
        "lizyml/config/loader.py",
        "params",
    ): "a config section name, not a LightGBM parameter; it has no alias",
    (
        "lizyml/calibration/isotonic.py",
        "seed",
    ): "the calibrator's own key, taken by documented precedence (H-0080)",
    (
        "lizyml/calibration/isotonic.py",
        "num_boost_round",
    ): "the calibrator's own key, excluded from canonicalisation",
    (
        "lizyml/calibration/isotonic.py",
        "validation_ratio",
    ): "the calibrator's own key, excluded from canonicalisation",
    (
        "lizyml/calibration/isotonic.py",
        "min_data_in_leaf_ratio",
    ): "the calibrator's own key, excluded from canonicalisation",
}


def _literal_reads(sources: dict[str, str]) -> set[tuple[str, str]]:
    """Return ``(path, key)`` for every literal-spelling read of a param dict."""
    found: set[tuple[str, str]] = set()

    for path, source in sources.items():
        tree = ast.parse(source)
        for node in ast.walk(tree):
            target = None
            key = None
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                target, key = node.value, node.slice.value
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"get", "pop", "setdefault"}
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                target, key = node.func.value, node.args[0].value

            if target is None or not isinstance(key, str):
                continue
            name = (
                target.id
                if isinstance(target, ast.Name)
                else target.attr
                if isinstance(target, ast.Attribute)
                else None
            )
            if name in _DICT_NAMES:
                found.add((path, key))
    return found


def _package_reads() -> set[tuple[str, str]]:
    sources = {
        str(path.relative_to(REPO)): path.read_text(encoding="utf-8")
        for path in sorted((REPO / "lizyml").rglob("*.py"))
    }
    return _literal_reads(sources)


def test_every_literal_parameter_read_is_declared() -> None:
    """A new one cannot arrive quietly.

    The defect this closes was a read that disagreed with the code that trained.
    Nothing about a read looks wrong locally, which is why the population has to
    be enumerated instead of reviewed.
    """
    undeclared = _package_reads() - set(_ALLOWED)
    assert not undeclared, (
        "literal-spelling reads of a caller-spelled parameter dict with no "
        f"entry in `_ALLOWED`: {sorted(undeclared)}"
    )


def test_no_declared_read_has_gone_away() -> None:
    """The allow-list is not allowed to rot either.

    An entry naming a read that no longer exists is a claim about code that is
    not there -- the stale half of the SSOT/derived pair (DC3).
    """
    stale = set(_ALLOWED) - _package_reads()
    assert not stale, f"declared reads that no longer exist: {sorted(stale)}"


def test_the_one_read_justified_by_having_no_alias_really_has_none() -> None:
    """`smart_params.py` reads `max_depth` under one spelling, and may.

    That is the only entry above whose reason is a fact about LightGBM rather
    than about LizyML's own code, so it is the only one that can go stale
    without anything in this repository changing.
    """
    from lizyml.estimators.lgbm.param_names import accepted_spellings

    assert sorted(accepted_spellings("max_depth")) == ["max_depth"], (
        "`max_depth` has gained an alias, so the read in smart_params.py is no "
        "longer safe under one spelling"
    )


#: Sources fed to the scan directly, so the scan is falsifiable rather than
#: merely present. Two are negative controls: a scan that reports everything
#: would pass every positive case and be worthless.
_HOSTILE: dict[str, tuple[str, set[tuple[str, str]]]] = {
    "subscript read": ("v = params['eta']\n", {("m.py", "eta")}),
    "get read": ("v = params.get('eta')\n", {("m.py", "eta")}),
    "pop read": ("v = user_params.pop('eta')\n", {("m.py", "eta")}),
    "attribute dict": ("v = self.params.get('eta')\n", {("m.py", "eta")}),
    "setdefault": ("params.setdefault('eta', 1)\n", {("m.py", "eta")}),
    "negative: not a param dict": ("v = row['eta']\n", set()),
    "negative: variable key": ("v = params[name]\n", set()),
}


@pytest.mark.parametrize("label", sorted(_HOSTILE))
def test_the_scan_detects_an_injected_read(label: str) -> None:
    source, expected = _HOSTILE[label]
    assert _literal_reads({"m.py": source}) == expected
