"""Count CI runs that a per-PR cancel-in-progress group would have cancelled.

Input: JSON list of runs (gh run list --json databaseId,event,headBranch,createdAt,updatedAt,status).
A run is "superseded in progress" when a later pull_request run on the same head
branch was created before this run finished.
"""

import collections
import datetime as dt
import json
import sys


def parse(x: str) -> dt.datetime:
    return dt.datetime.fromisoformat(x.replace("Z", "+00:00"))


runs = json.load(open(sys.argv[1]))
pr_runs = [r for r in runs if r["event"] == "pull_request" and r["status"] == "completed"]
by_branch: dict[str, list[dict]] = collections.defaultdict(list)
for r in pr_runs:
    by_branch[r["headBranch"]].append(r)

superseded = 0
for branch, items in by_branch.items():
    items.sort(key=lambda r: parse(r["createdAt"]))
    for i, r in enumerate(items[:-1]):
        nxt = items[i + 1]
        if parse(nxt["createdAt"]) < parse(r["updatedAt"]):
            superseded += 1

push_main = sum(1 for r in runs if r["event"] == "push")
first = min(parse(r["createdAt"]) for r in runs).date()
last = max(parse(r["createdAt"]) for r in runs).date()
print(f"window {first}..{last}: {len(runs)} runs, {len(pr_runs)} completed pull_request runs on {len(by_branch)} branches, {push_main} push runs")
print(f"superseded in progress: {superseded}/{len(pr_runs)}")
