#!/usr/bin/env bash
set -euo pipefail
S=/tmp/claude-1000/-home-rem-repos-LizyML/c356a2d3-b200-4f51-9b9e-a7f5cf0cb3eb/scratchpad
gh run list --repo nbx-liz/LizyML --workflow ci.yml --limit 20 --json databaseId,conclusion --jq '.[]|select(.conclusion=="success")|.databaseId' > "$S/runs.txt"
: > "$S/jobs.tsv"
while read -r r; do
  gh api "repos/nbx-liz/LizyML/actions/runs/$r/jobs" --paginate --jq '.jobs[]|select(.conclusion=="success")|"\(.name)\t\(.started_at)\t\(.completed_at)"' >> "$S/jobs.tsv"
done < "$S/runs.txt"
python3 "$S/durations.py" "$S/jobs.tsv"
