"""Diff a committed H-0138 oracle against a regenerated one, case by case."""

import gzip
import json
import sys
from collections import Counter

old = json.load(gzip.open(sys.argv[1]))
new = json.load(gzip.open(sys.argv[2]))
print("top keys equal:", old.keys() == new.keys())
meta_diff = {k: (old["meta"].get(k), new["meta"].get(k))
             for k in set(old["meta"]) | set(new["meta"])
             if old["meta"].get(k) != new["meta"].get(k)}
print("meta diff:", meta_diff)
for section in ("cases", "temporal_cases"):
    o, n = old[section], new[section]
    if isinstance(o, dict):
        okeys, nkeys = set(o), set(n)
        print(section, "ids equal:", okeys == nkeys, len(okeys))
        changed = sorted(k for k in okeys & nkeys if o[k] != n[k])
    else:
        print(section, "len", len(o), len(n))
        changed = [i for i, (a, b) in enumerate(zip(o, n)) if a != b]
    kinds = Counter(str(k)[0] for k in changed)
    print(section, "changed:", len(changed), dict(kinds), changed[:10])
    for k in changed[:2]:
        a, b = (o[k], n[k])
        if isinstance(a, dict):
            print("  ", k, "fields changed:", sorted(f for f in a if a.get(f) != b.get(f)))
