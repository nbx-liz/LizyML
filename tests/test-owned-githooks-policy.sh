#!/usr/bin/env bash
# Verify the repository-owned core.hooksPath integration, complementing the
# portable artifact-language and installer mutation suite.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

expected="$(cd "$ROOT/.githooks" && pwd -P)"
effective="$(git -C "$ROOT" rev-parse --path-format=absolute --git-path hooks)"
[ "$effective" = "$expected" ] \
    || fail "Git resolves hooks to $effective, not $expected"

entries="$(git -C "$ROOT" ls-files -s -- .githooks)"
for hook in commit-msg pre-commit pre-merge-commit pre-push; do
    mode="$(printf '%s\n' "$entries" | awk -v path=".githooks/$hook" '$4 == path { print $1 }')"
    [ "$mode" = "100755" ] || fail "$hook has index mode ${mode:-missing}, not 100755"
done

push_record="$TMP/push-record"
printf 'refs/heads/feature/control %040d refs/heads/main %040d\n' 1 0 >"$push_record"
set +e
push_out="$(git -C "$ROOT" hook run --to-stdin="$push_record" pre-push 2>&1)"
push_rc=$?
set -e
[ "$push_rc" -ne 0 ] || fail "protected push record was accepted"
case "$push_out" in
    *"BLOCKED:"*) ;;
    *) fail "protected push refusal did not come from the managed policy" ;;
esac

good_message="$TMP/good-message"
bad_message="$TMP/bad-message"
printf '%s\n' 'fix(test): English control' >"$good_message"
printf '%s\n' 'fix(test): Japanese control 日本語' >"$bad_message"
git -C "$ROOT" hook run commit-msg -- "$good_message"
set +e
message_out="$(git -C "$ROOT" hook run commit-msg -- "$bad_message" 2>&1)"
message_rc=$?
set -e
[ "$message_rc" -ne 0 ] || fail "Japanese commit message was accepted"
case "$message_out" in
    *"BLOCKED:"*) ;;
    *) fail "message refusal did not come from the managed policy" ;;
esac

fixture="$TMP/repo"
git init -q -b main "$fixture"
set +e
main_out="$(cd "$fixture" && "$ROOT/.githooks/pre-commit" 2>&1)"
main_rc=$?
set -e
[ "$main_rc" -ne 0 ] || fail "pre-commit accepted main"
case "$main_out" in
    *"BLOCKED:"*) ;;
    *) fail "main refusal did not come from the managed policy" ;;
esac

git -C "$fixture" checkout -q -b feature/control
mkdir -p "$fixture/.githooks"
printf '%s\n' 'canonical managed fixture' >"$fixture/.githooks/artifact-language.py"
git -C "$fixture" add .githooks/artifact-language.py
(cd "$fixture" && "$ROOT/.githooks/pre-commit")

echo "owned githooks policy: passed"
