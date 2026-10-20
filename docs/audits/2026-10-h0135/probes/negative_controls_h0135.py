"""H-0135 negative controls (acceptance criterion 8): each check is guarded.

Every control patches one H-0135 check back to a plausible wrong
implementation (most of them the pre-H-0135 behaviour). It then runs the tests
that guard that check, selected by ``-k``, and expects them to FAIL. The
patches are undone after each control, and the test modules are re-imported
for every run. A final run without any patch must pass, because a control is
"caught" only against that clean baseline. The exit status is 1 when a control
is not caught or the baseline fails.

Usage (from the repository root):
    uv run python docs/audits/2026-10-h0135/probes/negative_controls_h0135.py
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import lizyml.persistence._manifest as manifest  # noqa: E402
import lizyml.persistence.exporter as exporter  # noqa: E402
import lizyml.persistence.loader as loader  # noqa: E402

TESTS = "tests/test_persistence/test_h0135_acceptance.py"


def _permissive_manifest(mp: pytest.MonkeyPatch) -> None:
    """P1-P3 off: the H-0083 reading, where anything not a dict means 'no checksums'."""

    def declared(metadata: dict[str, Any], fv: int, path: Any) -> Any:
        cs = metadata.get("checksums")
        if not cs or not isinstance(cs, dict):
            return None
        return cs.get("files")

    mp.setattr(loader, "declared_files", declared)


def _no_format_rule(mp: pytest.MonkeyPatch) -> None:
    """P2 off: an absent manifest is legacy in every format."""
    mp.setattr(manifest, "LAST_FORMAT_WITHOUT_CHECKSUMS", 99)


def _no_required(mp: pytest.MonkeyPatch) -> None:
    """P3 off: required pickles need not be listed."""
    mp.setattr(manifest, "REQUIRED_FILES", ())


def _loose_digest(mp: pytest.MonkeyPatch) -> None:
    """P1 digest shape off: any string is a digest."""
    mp.setattr(manifest, "_DIGEST", re.compile(r".*"))


def _any_algorithm(mp: pytest.MonkeyPatch) -> None:
    """P1 algorithm check off: every algorithm name is accepted."""
    mp.setattr(manifest, "CHECKSUM_ALGORITHM", _Anything())


class _Anything(str):
    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False

    __hash__ = str.__hash__


def _context_when_present(mp: pytest.MonkeyPatch) -> None:
    """P4 off (#373): the context is read whenever the file is there."""

    def is_read(ctx_pkl: Path, files: Any, path: Any) -> bool:
        return ctx_pkl.exists()

    mp.setattr(loader, "_context_is_read", is_read)


def _context_by_lowercase_name(mp: pytest.MonkeyPatch) -> None:
    """P5 off: a manifest key declares a file by its lowercased base name."""
    real = loader._context_is_read

    def is_read(ctx_pkl: Path, files: Any, path: Any) -> bool:
        if files is not None:
            files = {Path(k).name.lower(): v for k, v in files.items()}
        return real(ctx_pkl, files, path)

    mp.setattr(loader, "_context_is_read", is_read)
    real_expected = loader._expected

    def expected(files: Any, name: str) -> Any:
        if files is not None:
            files = {Path(k).name.lower(): v for k, v in files.items()}
        return real_expected(files, name)

    mp.setattr(loader, "_expected", expected)


def _no_cleanup(mp: pytest.MonkeyPatch) -> None:
    """P6 off: the exporter leaves an earlier export's optional files."""
    mp.setattr(exporter, "_OPTIONAL_FILES", ())


def _cleanup_everything(mp: pytest.MonkeyPatch) -> None:
    """P6 too wide: the exporter clears every other file in the directory."""
    real = exporter.write_checked

    def write(path: Any, *args: Any, **kwargs: Any) -> Any:
        out = Path(path)
        if out.is_dir():
            for f in out.iterdir():
                if f.is_file():
                    f.unlink()
        return real(path, *args, **kwargs)

    # Model.export imports write_checked at call time, so this reaches it.
    mp.setattr(exporter, "write_checked", write)


def _read_pickles_first(mp: pytest.MonkeyPatch) -> None:
    """P7 off: the payloads are read before the manifest is checked."""
    real = loader.declared_files

    def declared(metadata: dict[str, Any], fv: int, path: Any) -> Any:
        src = Path(path)
        loader._verify_and_read_pkl(src / "fit_result.pkl", None)
        return real(metadata, fv, path)

    mp.setattr(loader, "declared_files", declared)


CONTROLS: dict[str, tuple[Callable[[pytest.MonkeyPatch], None], str]] = {
    "P1-P3 permissive (#374)": (_permissive_manifest, "malformed and 374"),
    "P2 absent allowed in v3/v4": (_no_format_rule, "absent_manifest_is_refused"),
    "P3 required not checked": (_no_required, "unlisted_required"),
    "P1 any digest string": (_loose_digest, "malformed and digest"),
    "P1 any algorithm": (_any_algorithm, "malformed and algorithm"),
    "P4 context read when present (#373)": (
        _context_when_present,
        "left_over or exact_name or declared_context_that_is_missing",
    ),
    "P5 names lowercased": (_context_by_lowercase_name, "exact_name"),
    "P6 no cleanup (#373)": (_no_cleanup, "reexport_without"),
    "P6 cleans user files": (_cleanup_everything, "keeps_files"),
    "P7 pickles read first": (
        _read_pickles_first,
        "refused_before_any_pickle or absent_manifest_is_refused or unlisted",
    ),
}


class _Patch:
    def __init__(self, apply: Callable[[pytest.MonkeyPatch], None]) -> None:
        self.apply = apply
        self.mp = pytest.MonkeyPatch()

    def pytest_configure(self, config: Any) -> None:
        self.apply(self.mp)

    def pytest_unconfigure(self, config: Any) -> None:
        self.mp.undo()


def _forget_test_modules() -> None:
    for module in [m for m in sys.modules if m == "tests" or m.startswith("tests.")]:
        del sys.modules[module]


def main() -> int:
    failures = 0
    for name, (apply, selection) in CONTROLS.items():
        _forget_test_modules()
        args = ["-q", "-p", "no:cacheprovider", "-x", "-k", selection, TESTS]
        code = pytest.main(args, plugins=[_Patch(apply)])
        caught = code == pytest.ExitCode.TESTS_FAILED
        # A patch that fails to apply, or a selection that matches nothing,
        # ends pytest with another exit code: report it, never "passed".
        verdict = "caught (tests failed)" if caught else f"NOT CAUGHT (pytest exit {int(code)})"
        print(f"CONTROL {name}: {verdict}")
        failures += not caught
    _forget_test_modules()
    baseline = pytest.main(["-q", "-p", "no:cacheprovider", TESTS])
    print(f"BASELINE without patches: exit {int(baseline)}")
    print(f"\n{len(CONTROLS) - failures}/{len(CONTROLS)} controls caught")
    return 1 if failures or baseline != pytest.ExitCode.OK else 0


if __name__ == "__main__":
    raise SystemExit(main())
