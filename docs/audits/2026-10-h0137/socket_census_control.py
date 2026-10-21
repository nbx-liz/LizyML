"""Positive control for socket_census.py: its logging cell sees the old fetches.

Needs the network (OpenML and raw.githubusercontent.com).

    uv run python docs/audits/2026-10-h0137/socket_census_control.py docs/audits/2026-10-h0137/socket_census.py
"""

import re
import sys
import tempfile
from pathlib import Path

src = Path(sys.argv[1]).read_text()
cell = re.search(r"LOG_CELL = r'''(.*?)'''", src, re.S).group(1)
ns: dict = {}
exec(cell, ns)

import pandas as pd
from sklearn.datasets import fetch_openml

fetch_openml("credit-g", version=1, as_frame=True, parser="auto", data_home=tempfile.mkdtemp())
print("after fetch_openml:", len(ns["_SOCKLOG"]), sorted({n for n, _ in ns["_SOCKLOG"]}))
n = len(ns["_SOCKLOG"])
pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv")
print("after read_csv URL:", len(ns["_SOCKLOG"]) - n, sorted({x for x, _ in ns["_SOCKLOG"][n:]}))
