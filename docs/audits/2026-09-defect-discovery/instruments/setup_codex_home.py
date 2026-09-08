"""Build a writable CODEX_HOME copy holding only what `codex exec` needs.

`~/.codex` carries credentials and session history. Codex needs a writable home,
so a copy is made rather than pointing it at the real one, and the copy holds
exactly two files. `cleanup_codex_home.py` deletes it after the run.

Shipped here rather than kept in a session scratchpad, which is where it lived
for the first twenty-four review rounds and where it was lost twice -- both
times recovered with `recover_from_transcript.py`, which is a recovery and not
a mechanism. A second loss is the point at which a note becomes a file.
"""

from __future__ import annotations

import pathlib
import shutil
import sys

SOURCE = pathlib.Path.home() / ".codex"
WANTED = ("auth.json", "config.toml")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: setup_codex_home.py <destination>", file=sys.stderr)
        return 2

    destination = pathlib.Path(sys.argv[1])
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    destination.chmod(0o700)

    for name in WANTED:
        origin = SOURCE / name
        if not origin.is_file():
            print(f"missing: {origin}", file=sys.stderr)
            return 1
        target = destination / name
        shutil.copyfile(origin, target)
        target.chmod(0o600)

    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
