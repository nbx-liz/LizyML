"""Rebuild the datasets the tutorial notebooks read (H-0137, #385).

Downloads each source listed in ``notebooks/data/manifest.toml``, writes the
vendored ``.csv.gz`` files, and checks that each tutorial's prepared frame,
built from the vendored file the way the notebook now builds it, equals the
frame the notebook built from the online source before H-0137. Prints the
sha256 of every file written, for the manifest.

This needs the network, so CI does not run it; it is the record of how the
files were made. ``tests/test_notebooks/test_notebook_data.py`` checks the
committed files against the manifest.

    uv run python scripts/vendor_notebook_data.py [--out DIR]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

ROOT = Path(__file__).resolve().parents[1]
DIAMONDS_URL = (
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"
)
JENA_URL = (
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    "jena_climate_2009_2016.csv.zip"
)
JENA_FORMAT = "%d.%m.%Y %H:%M:%S"
JENA_HOURS = 5000


def write_gz(frame: pd.DataFrame, path: Path) -> None:
    """``frame`` as CSV, gzip level 9 with mtime 0 (the bytes do not depend on when)."""
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as out:
        out.write(frame.to_csv(index=False).encode("utf-8"))
    path.write_bytes(buffer.getvalue())


def credit_g(out: Path) -> None:
    online = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    write_gz(online.frame, out / "credit-g.csv.gz")
    before = online.data.copy()
    for col in before.select_dtypes(include=["category"]).columns:
        before[col] = before[col].astype(str)
    before["credit_risk"] = (online.target == "bad").astype(int)
    after = pd.read_csv(out / "credit-g.csv.gz")
    after["credit_risk"] = (after.pop("class") == "bad").astype(int)
    pd.testing.assert_frame_equal(before, after)


def vehicle(out: Path) -> None:
    online = fetch_openml("vehicle", version=1, as_frame=True, parser="auto")
    write_gz(online.frame, out / "vehicle.csv.gz")

    def prepare(frame: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        frame.columns = [c.lower().replace(".", "_") for c in frame.columns]
        label_map = {c: i for i, c in enumerate(sorted(target.unique()))}
        frame["vehicle_class"] = target.map(label_map).astype(int)
        return frame

    before = prepare(online.data.copy(), online.target)
    after = pd.read_csv(out / "vehicle.csv.gz")
    after = prepare(after, after.pop("Class"))
    pd.testing.assert_frame_equal(before, after)


def diamonds(out: Path) -> None:
    online = pd.read_csv(DIAMONDS_URL)
    write_gz(online, out / "diamonds.csv.gz")
    pd.testing.assert_frame_equal(online, pd.read_csv(out / "diamonds.csv.gz"))


def _jena_tutorial_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Cells 5 and 6 of ``tutorial_time_series_lgbm.ipynb``."""
    raw = raw.copy()
    raw["Date Time"] = pd.to_datetime(raw["Date Time"], format=JENA_FORMAT)
    hourly = raw.set_index("Date Time").resample("1h").mean().dropna().reset_index()
    frame = hourly.tail(JENA_HOURS).reset_index(drop=True)
    frame["time_idx"] = range(len(frame))
    return frame.drop(columns=["Date Time"])


def jena(out: Path) -> None:
    """The 10-minute rows behind the last ``JENA_HOURS`` hourly rows.

    An hourly bin holds the rows from its hour on, so keeping every row from
    the first kept hour gives the same bins.
    """
    online = pd.read_csv(JENA_URL)
    stamps = pd.to_datetime(online["Date Time"], format=JENA_FORMAT)
    hourly = online.set_index(stamps).drop(columns=["Date Time"]).resample("1h").mean()
    start = hourly.dropna().index[-JENA_HOURS]
    write_gz(online[stamps >= start], out / "jena_climate_2016_tail.csv.gz")
    before = _jena_tutorial_frame(online)
    after = _jena_tutorial_frame(pd.read_csv(out / "jena_climate_2016_tail.csv.gz"))
    pd.testing.assert_frame_equal(before, after)
    print(f"jena: the rows from {start} on")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the tutorial datasets.")
    parser.add_argument("--out", type=Path, default=ROOT / "notebooks" / "data")
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    for build in (credit_g, vehicle, diamonds, jena):
        build(out)
    for path in sorted(out.glob("*.csv.gz")):
        print(path.name, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
