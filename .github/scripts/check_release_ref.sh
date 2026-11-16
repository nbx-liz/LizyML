#!/usr/bin/env bash
# Refuse a release run that is not on its own release tag (H-0146).
#
# The pypi / testpypi environments allow v* tags only, and the build checks
# out the run's ref, so the run must be on refs/tags/<TAG>. Inputs come through
# the environment:
#   TAG         the tag to publish: the dispatch input, or the published
#               release's tag_name
#   GITHUB_REF  the run's ref, set by GitHub Actions
set -euo pipefail

fail() {
  echo "::error::$1"
  exit 1
}

[ -n "${TAG:-}" ] || fail "TAG is not set"
[ -n "${GITHUB_REF:-}" ] || fail "GITHUB_REF is not set"

[[ "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] ||
  fail "tag '$TAG' is not of the form vX.Y.Z"

[ "$GITHUB_REF" = "refs/tags/${TAG}" ] ||
  fail "the run is on '$GITHUB_REF', not on refs/tags/${TAG}; dispatch with --ref ${TAG}"

echo "the run is on refs/tags/${TAG}"
