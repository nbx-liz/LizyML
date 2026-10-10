#!/usr/bin/env bash
# Decide whether the notebook index jobs run (H-0119 section 6).
#
# ci.yml runs this in the notebook-index-scope job and appends its one stdout
# line, run=true or run=false, to $GITHUB_OUTPUT. Diagnostics go to stderr.
# Inputs arrive through the environment:
#   EVENT     github.event_name
#   BASE_REF  the PR's base branch (empty outside a PR)
#   BASE_SHA  the PR's base commit
#   HEAD_SHA  the PR's head commit
#
# Anything but a PR to develop runs the jobs. A PR to develop runs them when a
# changed path is one of the index paths. The diff is read NUL-delimited (-z),
# because the default output quotes a path holding a non-ASCII character, a
# quote or a backslash ("notebooks/caf\303\251.ipynb"), and each NUL-delimited
# record is matched whole, so a newline inside a name can neither hide nor
# fake a match. --no-renames lists a rename as its deleted source and its
# added target: with rename detection (Git's default), --name-only names only
# the target, so a rename out of an index path would go unseen. The diff goes
# to a file first: a failing diff fails the script (set -e) instead of
# reporting that nothing changed.
set -euo pipefail

if [ "${EVENT:-}" != "pull_request" ] || [ "${BASE_REF:-}" != "develop" ]; then
  echo "Not a PR to develop: always run." >&2
  echo "run=true"
  exit 0
fi

[ -n "${BASE_SHA:-}" ] || { echo "::error::BASE_SHA is not set" >&2; exit 1; }
[ -n "${HEAD_SHA:-}" ] || { echo "::error::HEAD_SHA is not set" >&2; exit 1; }

changed=$(mktemp)
trap 'rm -f "$changed"' EXIT
git diff --name-only --no-renames -z "$BASE_SHA...$HEAD_SHA" >"$changed"
tr '\0' '\n' <"$changed" >&2

run=false
while IFS= read -r -d '' path; do
  case "$path" in
    notebooks/* | tests/test_notebooks/* | docs/examples.md | lizyml/_extras.py | \
      scripts/examples_index.py | .github/workflows/ci.yml)
      run=true
      ;;
  esac
done <"$changed"

if [ "$run" = false ]; then
  echo "No index path changed: the notebook index jobs are skipped." >&2
fi
echo "run=$run"
