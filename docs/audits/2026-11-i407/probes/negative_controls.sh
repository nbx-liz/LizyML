#!/usr/bin/env bash
# H-0146 acceptance 1: each control undoes one rule in a scratch copy of
# .github/ and records whether zizmor (offline) reports it.
# Usage: bash negative_controls.sh <repo> <scratch-dir>
set -euo pipefail
repo=$1
scratch=$2

run_control() {
  local name=$1 file=$2 from=$3 to=$4
  local dir="$scratch/$name"
  rm -rf "$dir"
  mkdir -p "$dir"
  cp -r "$repo/.github" "$dir/.github"
  python3 - "$dir/.github/workflows/$file" "$from" "$to" <<'PY'
import pathlib
import sys

path, old, new = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
text = path.read_text(encoding="utf-8")
assert old in text, old
path.write_text(text.replace(old, new, 1), encoding="utf-8")
PY
  set +e
  out=$(cd "$dir" && zizmor --offline --format plain . 2>/dev/null)
  code=$?
  set -e
  echo "== $name (exit $code)"
  printf '%s\n' "$out" | grep -E '^(error|warning|help|info)\[' | sort | uniq -c || echo "   no findings"
}

run_control unpinned-checkout ci.yml \
  "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1" "actions/checkout@v7"
run_control release-cache-on release.yml \
  "          enable-cache: false
" ""
run_control persisted-credentials-line-removed ci.yml \
  "          persist-credentials: false
" ""
run_control persisted-credentials-with-removed ci.yml \
  "        with:
          persist-credentials: false
" ""
run_control persisted-credentials-true ci.yml \
  "persist-credentials: false" "persist-credentials: true"
