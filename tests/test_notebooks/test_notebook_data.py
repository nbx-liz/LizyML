"""The vendored tutorial datasets match their manifest (H-0137, #385).

``notebooks/data/manifest.toml`` lists every file in ``notebooks/data/`` with
its sha256, the notebooks that read it, where it came from and the attribution
its licence requires. The notebooks read these files instead of downloading,
so a file that drifts from the record, a record without its file, or a record
missing any field must fail here, on every PR. Which files each notebook reads
is checked at run time, in both notebook layers (``network_guard.py``): a
substring of the source would not show that the file is read.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10; pytest itself depends on tomli there
    import tomli as tomllib

DATA = Path(__file__).resolve().parents[2] / "notebooks" / "data"
MANIFEST = DATA / "manifest.toml"
FIELDS = (
    "file",
    "sha256",
    "notebooks",
    "title",
    "creator",
    "source",
    "origin",
    "licence",
    "licence_url",
    "retrieved",
    "derivation",
    "notice",
)


def _entries(manifest: Path = MANIFEST) -> list[dict[str, Any]]:
    return list(tomllib.loads(manifest.read_text(encoding="utf-8"))["dataset"])


def manifest_problems(
    data: Path, entries: list[dict[str, Any]] | None = None
) -> list[str]:
    """Every way ``data`` disagrees with its ``manifest.toml`` (or ``entries``)."""
    problems: list[str] = []
    if entries is None:
        entries = _entries(data / "manifest.toml")
    listed = [entry.get("file") for entry in entries]
    if len(set(listed)) != len(listed):
        problems.append(f"a file is listed twice: {listed}")
    for entry in entries:
        if sorted(entry) != sorted(FIELDS):
            problems.append(f"{entry.get('file')}: fields {sorted(entry)}")
        if any(not entry.get(field) for field in FIELDS):
            problems.append(f"{entry.get('file')}: an empty field")
        path = data / str(entry.get("file"))
        if not path.is_file():
            problems.append(f"{path.name}: listed but missing")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry.get("sha256"):
            problems.append(f"{path.name}: sha256 {digest}")
        for name in entry.get("notebooks") or []:
            if not (data.parent / name).is_file():
                problems.append(f"{path.name}: notebook {name} missing")
    for path in sorted(data.iterdir()):
        if path.name != "manifest.toml" and path.name not in listed:
            problems.append(f"{path.name}: not in the manifest")
    return problems


def test_the_vendored_data_matches_its_manifest() -> None:
    assert manifest_problems(DATA) == []


def test_every_tutorial_that_read_remote_data_is_covered() -> None:
    covered = {name for entry in _entries() for name in entry["notebooks"]}
    assert covered == {
        "tutorial_binary_lgbm.ipynb",
        "tutorial_multiclass_lgbm.ipynb",
        "tutorial_regression_lgbm.ipynb",
        "tutorial_regression_tuning_lgbm.ipynb",
        "tutorial_time_series_lgbm.ipynb",
    }


def _copy(tmp_path: Path) -> Path:
    """``notebooks/`` (notebooks and data) under ``tmp_path``."""
    root = tmp_path / "notebooks"
    (root / "data").mkdir(parents=True)
    for path in DATA.iterdir():
        (root / "data" / path.name).write_bytes(path.read_bytes())
    for path in DATA.parent.glob("*.ipynb"):
        (root / path.name).write_bytes(path.read_bytes())
    return root / "data"


def _flip_last_byte(data: Path) -> str:
    path = data / "credit-g.csv.gz"
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(bytes(raw))
    return "credit-g.csv.gz: sha256"


def _add_unlisted(data: Path) -> str:
    (data / "extra.csv.gz").write_bytes(b"x")
    return "extra.csv.gz: not in the manifest"


def _drop_entry(data: Path) -> str:
    text = (data / "manifest.toml").read_text(encoding="utf-8")
    head, _, rest = text.partition('[[dataset]]\nfile = "vehicle.csv.gz"')
    _, _, tail = rest.partition("[[dataset]]")
    (data / "manifest.toml").write_text(head + "[[dataset]]" + tail, encoding="utf-8")
    return "vehicle.csv.gz: not in the manifest"


def _delete_file(data: Path) -> str:
    (data / "diamonds.csv.gz").unlink()
    return "diamonds.csv.gz: listed but missing"


def _unknown_notebook(data: Path) -> str:
    path = data / "manifest.toml"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("tutorial_time_series_lgbm", "tutorial_x"), "utf-8")
    return "jena_climate_2016_tail.csv.gz: notebook tutorial_x.ipynb missing"


@pytest.mark.parametrize(
    "mutate",
    [
        _flip_last_byte,
        _add_unlisted,
        _drop_entry,
        _delete_file,
        _unknown_notebook,
    ],
)
def test_each_drift_is_reported(mutate: Any, tmp_path: Path) -> None:
    data = _copy(tmp_path)
    assert manifest_problems(data) == []
    expected = mutate(data)
    problems = manifest_problems(data)
    assert any(problem.startswith(expected) for problem in problems), problems


@pytest.mark.parametrize("change", ["empty", "missing"])
@pytest.mark.parametrize("field", FIELDS)
def test_each_field_is_required(field: str, change: str) -> None:
    entries = _entries()
    diamonds = next(e for e in entries if e["file"] == "diamonds.csv.gz")
    if change == "empty":
        diamonds[field] = [] if field == "notebooks" else ""
    else:
        del diamonds[field]
    assert manifest_problems(DATA, entries) != []
