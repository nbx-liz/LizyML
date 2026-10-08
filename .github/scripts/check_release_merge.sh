#!/usr/bin/env bash
# Refuse to tag a release that is not a merge commit of develop (H-0117).
#
# auto-release.yml runs this before "Create tag". Inputs arrive through the
# environment, never through ${{ }} expansion inside the script:
#   HEAD_REF   the release PR's head branch
#   HEAD_REPO  the full name of the repository the head branch lives in
#   BASE_REPO  this repository's full name
#   MERGE_SHA  the PR's merge commit
set -euo pipefail

fail() {
  echo "::error::$1"
  exit 1
}

[ -n "${HEAD_REF:-}" ] || fail "HEAD_REF is not set"
[ -n "${HEAD_REPO:-}" ] || fail "HEAD_REPO is not set"
[ -n "${BASE_REPO:-}" ] || fail "BASE_REPO is not set"
[ -n "${MERGE_SHA:-}" ] || fail "MERGE_SHA is not set"

[ "$HEAD_REF" = "develop" ] ||
  fail "a release PR must come from 'develop', this one came from '$HEAD_REF' (CONTRIBUTING.md, Release)"
[ "$HEAD_REPO" = "$BASE_REPO" ] ||
  fail "a release PR must come from $BASE_REPO itself, this one came from $HEAD_REPO"

parents=$(git rev-list --parents -n 1 "$MERGE_SHA" 2>/dev/null) ||
  fail "merge commit $MERGE_SHA is not in this checkout"
read -r -a fields <<<"$parents"
count=$((${#fields[@]} - 1))
[ "$count" -eq 2 ] ||
  fail "merge commit $MERGE_SHA has $count parent(s); merge the release PR with 'Create a merge commit', not squash or rebase"

echo "release merge verified: $MERGE_SHA is a two-parent merge of develop"
