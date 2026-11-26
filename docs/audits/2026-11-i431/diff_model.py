"""Which parts of the model answers changed between two H-0138 oracles."""

import gzip
import json
import sys
from collections import Counter

old = json.load(gzip.open(sys.argv[1]))["cases"]
new = json.load(gzip.open(sys.argv[2]))["cases"]
fields: Counter[str] = Counter()
n_model = 0
max_pred_diff = 0.0
example = None
for a, b in zip(old, new):
    if "model" not in a:
        continue
    n_model += 1
    ma, mb = a["model"], b["model"]
    if not isinstance(ma, dict):
        fields["<scalar>"] += ma != mb
        continue
    for f in set(ma) | set(mb):
        if ma.get(f) != mb.get(f):
            fields[f] += 1
            if example is None:
                example = (a.get("id"), f, str(ma.get(f))[:200], str(mb.get(f))[:200])
    for f in ma:
        va, vb = ma.get(f), mb.get(f)
        if isinstance(va, list) and va and all(isinstance(x, (int, float)) for x in va):
            max_pred_diff = max(max_pred_diff, max(abs(x - y) for x, y in zip(va, vb)))
print("model cases:", n_model, "model keys:", sorted(old[0]["model"]) if isinstance(old[0].get("model"), dict) else type(old[0].get("model")))
print("changed fields:", dict(fields))
print("max |numeric list diff|:", max_pred_diff)
print("example:", example)
