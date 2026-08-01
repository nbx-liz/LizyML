#!/usr/bin/env bash
# Prove the three canonical Python files are exact exemptions, not a directory
# or prefix bypass in either aggregate Ruff or the staged-file hook.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROGUE_REL=".githooks/not-managed.py"
ROGUE="$ROOT/$ROGUE_REL"

# shellcheck source=../.githooks/protected-refs.sh
. "$ROOT/.githooks/protected-refs.sh"

[ ! -e "$ROGUE" ] || {
    echo "FAIL: mutation path already exists: $ROGUE_REL" >&2
    exit 1
}

cleanup() {
    git -C "$ROOT" reset -q -- "$ROGUE_REL" 2>/dev/null || true
    rm -f "$ROGUE"
}
trap cleanup EXIT

printf '%s\n' 'import os' >"$ROGUE"

set +e
aggregate_out="$(cd "$ROOT" && uv run --no-sync ruff check . 2>&1)"
aggregate_rc=$?
set -e
[ "$aggregate_rc" -ne 0 ] || {
    echo "FAIL: aggregate Ruff accepted $ROGUE_REL" >&2
    exit 1
}
case "$aggregate_out" in *F401*) ;; *) fail_reason="missing F401" ;; esac
case "$aggregate_out" in *"$ROGUE_REL"*) ;; *) fail_reason="${fail_reason:+$fail_reason; }missing path" ;; esac
[ -z "${fail_reason:-}" ] || {
    echo "FAIL: aggregate refusal was not the seeded F401 ($fail_reason)" >&2
    exit 1
}

git -C "$ROOT" add -f -- "$ROGUE_REL"
set +e
staged_out="$(cd "$ROOT" && .githooks/pre-commit 2>&1)"
staged_rc=$?
set -e
[ "$staged_rc" -ne 0 ] || {
    echo "FAIL: staged hook accepted $ROGUE_REL" >&2
    exit 1
}

current_ref="$(git -C "$ROOT" symbolic-ref --quiet HEAD || true)"
if [ -n "$current_ref" ] && protected_ref "$current_ref"; then
    # A real default-branch push checks out main/master as a branch. The
    # protected-ref refusal intentionally runs before staged Ruff, so assert
    # that fail-closed boundary here. Pull-request/detached and feature-branch
    # runs take the other arm and prove the staged-file Ruff routing itself.
    case "$staged_out" in
        *"BLOCKED: commit on protected branch '${current_ref#refs/heads/}'"*) ;;
        *)
            echo "FAIL: staged hook did not refuse protected $current_ref" >&2
            exit 1
            ;;
    esac
else
    unset fail_reason
    case "$staged_out" in *F401*) ;; *) fail_reason="missing F401" ;; esac
    case "$staged_out" in *"$ROGUE_REL"*) ;; *) fail_reason="${fail_reason:+$fail_reason; }missing path" ;; esac
    [ -z "${fail_reason:-}" ] || {
        echo "FAIL: staged refusal was not the seeded F401 ($fail_reason)" >&2
        exit 1
    }
fi

echo "Ruff policy boundary mutations: passed"
