#!/usr/bin/env bash
# sdist and wheel of a base commit and of HEAD, each built from a clean worktree
# (files that are not tracked never reach the build): sizes and notebooks/ entries.
#   bash docs/audits/2026-10-h0137/dist_contents.sh <base-commit> > docs/audits/2026-10-h0137/dist_contents.out
set -euo pipefail
cd "$(dirname "$0")/../../.."
BASE=$(git rev-parse "${1:?base commit}")
HEAD_SHA=$(git rev-parse HEAD)
WORK=$(mktemp -d)
cleanup() {
  for side in base head; do
    git worktree remove --force "$WORK/$side" >/dev/null 2>&1 || true
  done
  rm -rf "$WORK"
}
trap cleanup EXIT
git worktree add --detach "$WORK/base" "$BASE" >/dev/null 2>&1
git worktree add --detach "$WORK/head" "$HEAD_SHA" >/dev/null 2>&1
for side in base head; do
  sha=$([ "$side" = base ] && echo "$BASE" || echo "$HEAD_SHA")
  out="$WORK/dist-$side"
  (cd "$WORK/$side" && UV_CACHE_DIR=${UV_CACHE_DIR:-$TMPDIR/uv-cache} uv build -q -o "$out" >/dev/null 2>&1)
  echo "== $side (${sha:0:7})"
  for f in "$out"/*.tar.gz "$out"/*.whl; do
    name=$(basename "$f" | sed -E 's/^lizyml-[^-]+(-py3-none-any\.whl|\.tar\.gz)$/lizyml-<version>\1/')
    echo "$name $(stat -c %s "$f") bytes"
  done
  echo "sdist notebooks/data entries:"
  tar tzf "$out"/*.tar.gz | sed -E 's#^[^/]+/##' | grep '^notebooks/data/' || echo "  (none)"
  echo "wheel notebooks/ entries: $(python3 -m zipfile -l "$out"/*.whl | grep -c 'notebooks/' || true)"
done
