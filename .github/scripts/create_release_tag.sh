#!/usr/bin/env bash
# Create the release tag on the merge commit, idempotently (H-0117).
#
# A re-run after a later step failed (GitHub Release, PyPI dispatch) must be
# able to finish the same release, so a tag that already names MERGE_SHA is
# reused and pushed again. A tag that names any other commit is refused.
# Inputs come through the environment:
#   TAG        the release tag, vX.Y.Z
#   MERGE_SHA  the release PR's merge commit
set -euo pipefail

fail() {
  echo "::error::$1"
  exit 1
}

[ -n "${TAG:-}" ] || fail "TAG is not set"
[ -n "${MERGE_SHA:-}" ] || fail "MERGE_SHA is not set"

merge=$(git rev-parse --verify --quiet "${MERGE_SHA}^{commit}") ||
  fail "merge commit $MERGE_SHA is not in this checkout"

if existing=$(git rev-parse --verify --quiet "refs/tags/${TAG}^{commit}"); then
  [ "$existing" = "$merge" ] ||
    fail "tag $TAG already exists on $existing, not on the merge commit $merge"
  echo "tag $TAG already names $merge; reusing it"
else
  git tag "$TAG" "$merge"
  echo "tagged $merge as $TAG"
fi
git push origin "refs/tags/${TAG}"
