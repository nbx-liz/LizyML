import collections
import json
import os

counts: collections.Counter = collections.Counter()
examples: dict = {}
for line in open(os.environ.get("PR6_FIRING_LOG", "/tmp/claude-1000/pr6_firing.jsonl"), encoding="utf-8"):
    d = json.loads(line)
    if d["kind"] == "TOTALS":
        print("TOTALS", d)
        continue
    key = (d["kind"], d["ok_before"])
    counts[key] += 1
    examples.setdefault(key, []).append((d["test"][:100], d["detail"]))
for key, n in counts.items():
    print(key, n)
    seen = set()
    for test, detail in examples[key]:
        sig = test.split("::")[0] + str(detail)[:60]
        if sig in seen:
            continue
        seen.add(sig)
        print("   ", test, detail)
        if len(seen) >= 8:
            break
