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
    -k "guard or network or twin or uninstall or entry_point or data_files or open or classifies or address or accounted or probed" \
    2>&1 | tail -1
  cp "$BACKUP" "$F"
}
sed -i 's/            raise NetworkRefused(f"{MESSAGE}: {call}")/            return None/' "$F"; run "M1 no refusal"
sed -i 's/        if self.attempts:/        if False:/' "$F"; run "M2 network check never fails"
sed -i 's/            self.attempts.append(call)/            pass/' "$F"; run "M3 no record"
sed -i 's/                delattr(owner, name)/                pass/' "$F"; run "M4 uninstall leaves inherited patch"
sed -i 's/^    "sendmsg": _ADDRESS,//' "$F"; run "M5 sendmsg not refused"
sed -i 's/^    "getnameinfo": _RESOLVES,//' "$F"; run "M6 getnameinfo not refused"
sed -i 's/                    self.reads.add(path.name)/                    pass/' "$F"; run "M7 reads not recorded"
sed -i 's/        if self.reads != set(expected):/        if False:/' "$F"; run "M8 reads never compared"
sed -i 's/^    "getfqdn": _RESOLVES,//' "$F"; run "M9 getfqdn not refused (round 2 finding)"
sed -i 's/^    "bind": _ADDRESS,//' "$F"; run "M12 bind not refused (round 3 finding)"
sed -i 's/ and not _WRITE_MODES \& set(mode):/:/' "$F"; run "M10 write opens recorded as reads (round 2 finding)"
sed -i 's/^    "accept": _BOUND,//' "$F"; run "M11 a public method left unclassified"
sed -i 's/^    "bind": _ADDRESS,//; s/^    "accept": _BOUND,/    "accept": _BOUND,\n    "bind": _BOUND,/' "$F"; run "M13 bind classified as not refused (the round-3 shape)"
sed -i 's/^    "inet_aton": _CONVERTS,//; s/^    "getservbyport": _NAME_DB,/    "getservbyport": _NAME_DB,\n    "inet_aton": _NAME_DB,/' "$F"; run "M14 a local function moved to the refused side"
run "M0 original"
