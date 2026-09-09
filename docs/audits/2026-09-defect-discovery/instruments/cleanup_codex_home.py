"""Delete the writable CODEX_HOME copy. Run after every Codex invocation."""

from __future__ import annotations

import pathlib
import shutil
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: cleanup_codex_home.py <destination>", file=sys.stderr)
        return 2

    destination = pathlib.Path(sys.argv[1]).resolve()
    home = pathlib.Path.home().resolve()
    if destination == home or home in destination.parents:
        # Never let a mistyped path reach the real ~/.codex.
        print(f"refusing to delete inside the home directory: {destination}")
        return 1

    if destination.exists():
        shutil.rmtree(destination)
        print(f"deleted {destination}")
    else:
        print(f"nothing to delete at {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
