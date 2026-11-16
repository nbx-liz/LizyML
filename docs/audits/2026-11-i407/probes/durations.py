import collections
import datetime as dt
import sys


def parse(x: str) -> dt.datetime:
    return dt.datetime.fromisoformat(x.replace("Z", "+00:00"))


groups: dict[str, list[float]] = collections.defaultdict(list)
for line in open(sys.argv[1]):
    name, start, end = line.rstrip("\n").split("\t")
    key = name
    if name.startswith(("Notebook index (", "Extras registry")) and "gate" not in name and "matrix" not in name:
        key = name.split(" (")[0] + " (*)"
    groups[key].append((parse(end) - parse(start)).total_seconds() / 60)

for key, vals in sorted(groups.items()):
    vals.sort()
    print(f"{vals[-1]:6.1f} max {vals[len(vals) // 2]:6.1f} med n={len(vals):3d} {key}")
