"""H-0135 (#373, #374): load reads only what ``checksums`` declares.

Acceptance criteria 1-5 and 7 of H-0135:

1. #373: a re-export into a directory with an earlier export's optional files
   removes them, and a left-over context is not read.
2. #374: a list-shaped ``checksums`` in an untuned artifact is refused.
3. P1: every malformed ``checksums`` is refused, in every format.
4. P2: an absent ``checksums`` loads in format 1-2 only.
5. P3-P5: required pickles must be listed; the four states of
   ``analysis_context.pkl``; exact names; symlinks.
7. P6: the re-export removes only the export's own optional files.

Every refusal is checked to come before any pickle is read: the trap makes
reading or unpickling a payload fail the test.
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.tuning_codec import HISTORY_FILE
from lizyml.persistence import load as persistence_load
from lizyml.persistence import loader
from tests._helpers import make_config, make_regression_df
from tests.test_persistence._artifact_versions import (
    resign,
    to_v1_shape,
    to_v2_shape,
    to_v3_shape,
)

CONTEXT = "analysis_context.pkl"
DF = make_regression_df(n=120, seed=0)
SHAPES: dict[str, Callable[[Path], None]] = {
    "v1": to_v1_shape,
    "v2": to_v2_shape,
    "v3": to_v3_shape,
    "v4": lambda _: None,
}


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(scope="module")
def plain(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An untuned regression artifact with an analysis context."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Model(make_config("regression", n_estimators=5, n_splits=2))
        model.fit(data=DF)
        return model.export(tmp_path_factory.mktemp("h0135") / "plain")


@pytest.fixture(scope="module")
def tuned(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tuned regression artifact with a tuning history."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Model(
            make_config("regression", n_estimators=5, n_splits=2, tuning_n_trials=1)
        )
        model.tune(data=DF)
        model.fit(data=DF)
        return model.export(tmp_path_factory.mktemp("h0135") / "tuned")


def _copy(artifact: Path, dest: Path) -> Path:
    dest.mkdir()
    for f in artifact.iterdir():
        (dest / f.name).write_bytes(f.read_bytes())
    return dest


def _meta(artifact: Path) -> dict[str, Any]:
    return json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))


def _write_meta(artifact: Path, meta: dict[str, Any]) -> None:
    (artifact / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")


def _without_context(artifact: Path) -> Path:
    """The artifact as a writer without a context writes it: no file, no entry."""
    (artifact / CONTEXT).unlink()
    meta = _meta(artifact)
    del meta["checksums"]["files"][CONTEXT]
    _write_meta(artifact, meta)
    return artifact


def _trap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if any payload is read or unpickled."""

    def no_pickle(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a pickle was read before the artifact was refused")

    monkeypatch.setattr(loader, "_verify_and_read_pkl", no_pickle)
    monkeypatch.setattr(loader.joblib, "load", no_pickle)


def _refused(artifact: Path, monkeypatch: pytest.MonkeyPatch) -> LizyMLError:
    """The refusal of *artifact*, which must come before any pickle is read."""
    _trap(monkeypatch)
    with pytest.raises(LizyMLError) as info:
        Model.load(artifact)
    assert info.value.code == ErrorCode.DESERIALIZATION_FAILED
    monkeypatch.undo()
    return info.value


# ---------------------------------------------------------------------------
# Criterion 1: #373 and the history file
# ---------------------------------------------------------------------------


def test_a_reexport_without_context_removes_the_old_one(
    plain: Path, tmp_path: Path
) -> None:
    reused = _copy(plain, tmp_path / "reused")
    without = Model.load(_without_context(_copy(plain, tmp_path / "without")))
    assert (reused / CONTEXT).exists()
    without.export(reused)
    assert not (reused / CONTEXT).exists()
    with pytest.raises(LizyMLError) as info:
        Model.load(reused).residuals()
    assert info.value.code == ErrorCode.MODEL_NOT_FIT


def test_a_left_over_context_is_ignored_and_logged(
    plain: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The reader alone closes #373: an undeclared context is never read."""
    artifact = _without_context(_copy(plain, tmp_path / "a"))
    (artifact / CONTEXT).write_bytes((plain / CONTEXT).read_bytes())
    with caplog.at_level(logging.INFO, logger="lizyml.persistence.loader"):
        model = Model.load(artifact)
    assert any(CONTEXT in r.getMessage() for r in caplog.records)
    with pytest.raises(LizyMLError) as info:
        model.residuals()
    assert info.value.code == ErrorCode.MODEL_NOT_FIT


def test_a_reexport_without_history_removes_the_old_one(
    plain: Path, tuned: Path, tmp_path: Path
) -> None:
    reused = _copy(tuned, tmp_path / "reused")
    assert (reused / HISTORY_FILE).exists()
    Model.load(plain).export(reused)
    assert not (reused / HISTORY_FILE).exists()
    assert HISTORY_FILE not in _meta(reused)["checksums"]["files"]


# ---------------------------------------------------------------------------
# Criteria 2 and 3: #374 and every malformed manifest, in every format
# ---------------------------------------------------------------------------

_DIGEST = "0" * 64


def _files(meta: dict[str, Any]) -> dict[str, str]:
    return dict(meta["checksums"]["files"])


MALFORMED: dict[str, tuple[Callable[[dict[str, Any]], Any], str]] = {
    # name: (the checksums value built from the valid metadata, reason)
    "empty list (#374)": (lambda m: [], "checksums_malformed"),
    "list (#374)": (lambda m: ["x"], "checksums_malformed"),
    "null": (lambda m: None, "checksums_malformed"),
    "number": (lambda m: 1, "checksums_malformed"),
    "boolean": (lambda m: True, "checksums_malformed"),
    "string": (lambda m: "sha256", "checksums_malformed"),
    "empty object": (lambda m: {}, "checksums_malformed"),
    "no files": (lambda m: {"algorithm": "sha256"}, "checksums_malformed"),
    "no algorithm": (lambda m: {"files": _files(m)}, "checksums_malformed"),
    "extra key": (
        lambda m: {"algorithm": "sha256", "files": _files(m), "x": 1},
        "checksums_malformed",
    ),
    "files is a list": (
        lambda m: {"algorithm": "sha256", "files": list(_files(m))},
        "checksums_malformed",
    ),
    "algorithm is a number": (
        lambda m: {"algorithm": 256, "files": _files(m)},
        "checksums_malformed",
    ),
    "algorithm SHA256": (
        lambda m: {"algorithm": "SHA256", "files": _files(m)},
        "checksums_algorithm",
    ),
    "algorithm md5": (
        lambda m: {"algorithm": "md5", "files": _files(m)},
        "checksums_algorithm",
    ),
    "digest is a number": (
        lambda m: {"algorithm": "sha256", "files": {**_files(m), "x": 1}},
        "checksums_malformed",
    ),
    "digest is short": (
        lambda m: {"algorithm": "sha256", "files": {**_files(m), "x": "ab"}},
        "checksums_malformed",
    ),
    "digest is uppercase": (
        lambda m: {
            "algorithm": "sha256",
            "files": {k: v.upper() for k, v in _files(m).items()},
        },
        "checksums_malformed",
    ),
    "digest is not hex": (
        lambda m: {"algorithm": "sha256", "files": {**_files(m), "x": "g" * 64}},
        "checksums_malformed",
    ),
}


@pytest.mark.parametrize("version", sorted(SHAPES))
@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_a_malformed_manifest_is_refused_before_any_pickle(
    case: str,
    version: str,
    plain: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _copy(plain, tmp_path / "a")
    SHAPES[version](artifact)
    build, reason = MALFORMED[case]
    meta = _meta(artifact)
    meta["checksums"] = build(meta)
    _write_meta(artifact, meta)
    assert _refused(artifact, monkeypatch).context["reason"] == reason


@pytest.mark.parametrize("version", sorted(SHAPES))
def test_a_well_formed_manifest_loads_in_every_format(
    version: str, plain: Path, tmp_path: Path
) -> None:
    """The control of the two tests above and below."""
    artifact = _copy(plain, tmp_path / "a")
    SHAPES[version](artifact)
    _, _, _, context = persistence_load(artifact)
    assert context is not None


# ---------------------------------------------------------------------------
# Criterion 4: P2, an absent manifest
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_an_absent_manifest_loads_in_format_1_and_2(
    version: str, plain: Path, tmp_path: Path
) -> None:
    artifact = _copy(plain, tmp_path / "a")
    SHAPES[version](artifact)
    meta = _meta(artifact)
    del meta["checksums"]
    _write_meta(artifact, meta)
    _, _, _, context = persistence_load(artifact)
    assert context is not None  # read when present, as before H-0083


@pytest.mark.parametrize("version", ["v3", "v4"])
def test_an_absent_manifest_is_refused_in_format_3_and_4(
    version: str, plain: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _copy(plain, tmp_path / "a")
    SHAPES[version](artifact)
    meta = _meta(artifact)
    del meta["checksums"]
    _write_meta(artifact, meta)
    error = _refused(artifact, monkeypatch)
    assert error.context["reason"] == "checksums_missing"


# ---------------------------------------------------------------------------
# Criterion 5: P3, P4, P5
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", sorted(SHAPES))
@pytest.mark.parametrize("name", ["fit_result.pkl", "refit_model.pkl"])
def test_an_unlisted_required_pickle_is_refused(
    name: str,
    version: str,
    plain: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _copy(plain, tmp_path / "a")
    SHAPES[version](artifact)
    meta = _meta(artifact)
    digest = meta["checksums"]["files"].pop(name)
    # A path-like or differently cased key does not declare the file (P5).
    meta["checksums"]["files"][f"../{name}"] = digest
    meta["checksums"]["files"][name.upper()] = digest
    _write_meta(artifact, meta)
    error = _refused(artifact, monkeypatch)
    assert error.context["reason"] == "required_file_unlisted"
    assert error.context["file"] == name


def test_a_declared_context_that_is_missing_is_refused(
    plain: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _copy(plain, tmp_path / "a")
    (artifact / CONTEXT).unlink()
    error = _refused(artifact, monkeypatch)
    assert error.context["reason"] == "declared_file_missing"
    assert error.context["file"] == CONTEXT


@pytest.mark.parametrize(
    "key", ["Analysis_Context.pkl", "../analysis_context.pkl", "./analysis_context.pkl"]
)
def test_only_the_exact_name_declares_the_context(
    key: str, plain: Path, tmp_path: Path
) -> None:
    artifact = _copy(plain, tmp_path / "a")
    meta = _meta(artifact)
    meta["checksums"]["files"][key] = meta["checksums"]["files"].pop(CONTEXT)
    _write_meta(artifact, meta)
    _, _, _, context = persistence_load(artifact)
    assert context is None


def test_a_declared_and_present_context_is_read(plain: Path, tmp_path: Path) -> None:
    _, _, _, context = persistence_load(_copy(plain, tmp_path / "a"))
    assert context is not None


def test_a_declared_context_must_match_its_digest(plain: Path, tmp_path: Path) -> None:
    artifact = _copy(plain, tmp_path / "a")
    meta = _meta(artifact)
    meta["checksums"]["files"][CONTEXT] = _DIGEST
    _write_meta(artifact, meta)
    with pytest.raises(LizyMLError) as info:
        Model.load(artifact)
    assert info.value.code == ErrorCode.DESERIALIZATION_FAILED
    assert info.value.context["file"] == CONTEXT


@pytest.mark.parametrize("name", ["fit_result.pkl", CONTEXT])
def test_a_fixed_name_symlink_is_read_when_its_bytes_match(
    name: str, plain: Path, tmp_path: Path
) -> None:
    artifact = _copy(plain, tmp_path / "a")
    elsewhere = tmp_path / "elsewhere.bin"
    elsewhere.write_bytes((artifact / name).read_bytes())
    (artifact / name).unlink()
    (artifact / name).symlink_to(elsewhere)
    fit_result, _, _, context = persistence_load(artifact)
    assert fit_result is not None and context is not None


def test_a_fixed_name_symlink_to_other_bytes_is_refused(
    plain: Path, tmp_path: Path
) -> None:
    artifact = _copy(plain, tmp_path / "a")
    elsewhere = tmp_path / "elsewhere.bin"
    elsewhere.write_bytes((artifact / "refit_model.pkl").read_bytes())
    (artifact / "fit_result.pkl").unlink()
    (artifact / "fit_result.pkl").symlink_to(elsewhere)
    with pytest.raises(LizyMLError) as info:
        Model.load(artifact)
    assert info.value.context["file"] == "fit_result.pkl"


def test_resign_reproduces_the_exporter_digests(plain: Path, tmp_path: Path) -> None:
    """The helper the rewritten tests use writes what the exporter writes."""
    artifact = _copy(plain, tmp_path / "a")
    before = _meta(artifact)["checksums"]
    resign(artifact)
    assert _meta(artifact)["checksums"] == before
    assert (
        before["files"][CONTEXT]
        == hashlib.sha256((artifact / CONTEXT).read_bytes()).hexdigest()
    )


# ---------------------------------------------------------------------------
# Criterion 7: P6 removes only the export's own optional files
# ---------------------------------------------------------------------------


def test_a_reexport_keeps_files_it_does_not_own(plain: Path, tmp_path: Path) -> None:
    reused = _copy(plain, tmp_path / "reused")
    (reused / "notes.txt").write_text("mine", encoding="utf-8")
    (reused / "Analysis_Context.pkl").write_bytes(b"also mine")
    Model.load(_without_context(_copy(plain, tmp_path / "w"))).export(reused)
    assert (reused / "notes.txt").read_text(encoding="utf-8") == "mine"
    assert (reused / "Analysis_Context.pkl").read_bytes() == b"also mine"
    assert not (reused / CONTEXT).exists()


def test_a_reexport_with_context_rewrites_it(plain: Path, tmp_path: Path) -> None:
    reused = _copy(plain, tmp_path / "reused")
    Model.load(plain).export(reused)
    assert CONTEXT in _meta(reused)["checksums"]["files"]
    _, _, _, context = persistence_load(reused)
    assert context is not None
