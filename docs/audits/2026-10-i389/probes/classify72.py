"""Classify the 3.1.0 temporal cases that b297de4 fails at transform."""

from __future__ import annotations

import gzip
import json
import sys
from collections import Counter

GEN = "/tmp/claude-1000/i389-gen"
DATA = "/home/rem/repos/LizyML/tests/test_features/data"


def load(path: str) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


o310 = {c["id"]: c["outcome"] for c in load(f"{GEN}/oracle-3.1.0-a.json.gz")["temporal_cases"]}
o301 = {c["id"]: c["outcome"] for c in load(f"{DATA}/h0138_oracle_pandas-3.0.1.json.gz")["temporal_cases"]}
ov301 = {c["id"]: c for c in load(f"{DATA}/h0139_overlay_pandas-3.0.1.json.gz")["changes"]}
ov310 = {c["id"]: c for c in load(f"{GEN}/overlay-3.1.0-a.json.gz")["changes"]}

t72 = sorted(cid for cid, out in o310.items() if out.get("fails_at") == "transform")
print("transform failures in 3.1.0 oracle:", len(t72))
print("in 3.0.1 oracle at all:", sum(cid in o301 for cid in t72))
grid = Counter()
for cid in t72:
    old = o301.get(cid)
    old_shape = "absent" if old is None else old.get("fails_at", "ok")
    in_ov = ov301[cid]["rule"] if cid in ov301 else "-"
    grid[(old_shape, in_ov, o310[cid].get("exception"))] += 1
for key, n in sorted(grid.items()):
    print(key, n)
print("3.0.1 transform failures:", sum(1 for o in o301.values() if o.get("fails_at") == "transform"))
print("sample ids:", t72[:12])
# 3.1.0 vs 3.0.1 overlay membership
print("overlay ids only in 3.0.1:", sorted(set(ov301) - set(ov310))[:20], len(set(ov301) - set(ov310)))
print("overlay ids only in 3.1.0:", sorted(set(ov310) - set(ov301))[:20], len(set(ov310) - set(ov301)))
json.dump(t72, sys.stdout if len(sys.argv) < 2 else open(sys.argv[1], "w"))
