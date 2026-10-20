#!/usr/bin/env bash
# Mutants of network_guard.py: each must make the fast guard tests fail.
#   bash docs/audits/2026-10-h0137/mutate_guard.sh > docs/audits/2026-10-h0137/mutate_guard.out
set -u
cd "$(dirname "$0")/../../.."
F=tests/test_notebooks/network_guard.py
BACKUP=$(mktemp)
cp "$F" "$BACKUP"
trap 'cp "$BACKUP" "$F"' EXIT
run() {
  echo "== $1"
  UV_CACHE_DIR=$TMPDIR/uv-cache uv run pytest -q -p no:cacheprovider \
    tests/test_notebooks/test_index_recording.py -m "not slow" \
    -k "guard or network or twin or uninstall or entry_point or data_files" 2>&1 | tail -1
  cp "$BACKUP" "$F"
}
sed -i 's/            raise NetworkRefused(f"{MESSAGE}: {call}")/            return None/' "$F"; run "M1 no refusal"
sed -i 's/        if self.attempts:/        if False:/' "$F"; run "M2 network check never fails"
sed -i 's/            self.attempts.append(call)/            pass/' "$F"; run "M3 no record"
sed -i 's/                delattr(owner, name)/                pass/' "$F"; run "M4 uninstall leaves inherited patch"
sed -i 's/SOCKET_METHODS = ("connect", "connect_ex", "sendto", "sendmsg")/SOCKET_METHODS = ("connect", "connect_ex", "sendto")/' "$F"; run "M5 sendmsg not refused"
sed -i 's/    "getnameinfo",//' "$F"; run "M6 getnameinfo not refused"
sed -i 's/                    self.reads.add(path.name)/                    pass/' "$F"; run "M7 reads not recorded"
sed -i 's/        if self.reads != set(expected):/        if False:/' "$F"; run "M8 reads never compared"
run "M0 original"
