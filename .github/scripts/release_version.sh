#!/usr/bin/env bash
# Read the release version from a release PR title (H-0117).
#
# The whole title must be exactly `release: vX.Y.Z` (CONTRIBUTING.md, Release),
# so a suffix such as `-rc1`, a second version or trailing text is refused
# instead of yielding a tag the title did not name. Inputs come through the
# environment:
#   TITLE          the release PR's title
#   GITHUB_OUTPUT  the step-output file; receives tag=vX.Y.Z and version=X.Y.Z
set -euo pipefail

fail() {
  echo "::error::$1"
  exit 1
}

[ -n "${GITHUB_OUTPUT:-}" ] || fail "GITHUB_OUTPUT is not set"
title="${TITLE:-}"
pattern='^release: (v([0-9]+\.[0-9]+\.[0-9]+))$'
[[ "$title" =~ $pattern ]] ||
  fail "a release PR title must be exactly 'release: vX.Y.Z'; got '$title'"

tag="${BASH_REMATCH[1]}"
version="${BASH_REMATCH[2]}"
{
  echo "tag=$tag"
  echo "version=$version"
} >>"$GITHUB_OUTPUT"
echo "release version: $tag"
