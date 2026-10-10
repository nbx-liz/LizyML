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
# changed path matches one of the index paths. The diff is read NUL-delimited
# (-z), because the default output quotes a path holding a non-ASCII
# character, a quote or a backslash ("notebooks/caf\303\251.ipynb"), and a
# quoted path would not match. Turning each NUL into a newline can only split
# a path whose name holds a newline; every path still starts a line, so a
# split can add a spurious match but never hide one. A failing diff fails the
# script instead of reporting that nothing changed.
set -euo pipefail

if [ "${EVENT:-}" != "pull_request" ] || [ "${BASE_REF:-}" != "develop" ]; then
  echo "Not a PR to develop: always run." >&2
  echo "run=true"
  exit 0
fi

[ -n "${BASE_SHA:-}" ] || { echo "::error::BASE_SHA is not set" >&2; exit 1; }
[ -n "${HEAD_SHA:-}" ] || { echo "::error::HEAD_SHA is not set" >&2; exit 1; }

changed=$(git diff --name-only -z "$BASE_SHA...$HEAD_SHA" | tr '\0' '\n')
printf '%s\n' "$changed" >&2
pattern='^(notebooks/|tests/test_notebooks/|docs/examples\.md$|lizyml/_extras\.py$|scripts/examples_index\.py$|\.github/workflows/ci\.yml$)'
# A here-string, not a pipe: an early-exiting grep -q on a pipe would kill the
# writer with SIGPIPE, and pipefail would turn the match into a failure.
if grep -Eq "$pattern" <<<"$changed"; then
  echo "run=true"
else
  echo "No index path changed: the notebook index jobs are skipped." >&2
  echo "run=false"
fi
