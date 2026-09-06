#!/usr/bin/env python3
"""Replay Write/Edit tool calls out of a Claude Code transcript to recover lost files.

Written after the second loss of /tmp/lizyml-discovery-plan (2026-09-03 and
2026-09-05).  The first loss was recovered by an ad-hoc script that itself lived
in /tmp and is also gone; this one lives beside the transcripts it reads, so it
survives the thing it recovers from.  See the `discovery-plan-2026-09` memory.

Stateless: reads a transcript, writes files, keeps nothing between runs.

Usage
-----
    # what files does this transcript contain?
    python3 recover_from_transcript.py <session>.jsonl --list

    # restore everything it wrote under /tmp/lizyml-discovery-plan
    python3 recover_from_transcript.py <session>.jsonl \\
        --prefix /tmp/lizyml-discovery-plan --out ./recovered

    # search assistant text and tool results too (script-generated files are NOT
    # in Write calls -- their content only survives if it was printed)
    python3 recover_from_transcript.py <session>.jsonl --grep 'clauses added'

Limits
------
Only files created through the Write/Edit tools are recoverable.  Anything a
script wrote to disk exists in the transcript only where its content happened to
be echoed into a tool result.  --grep is how you find those; there is no
automatic recovery for them.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Iterator
from typing import Any


def iter_records(path: pathlib.Path) -> Iterator[tuple[int, dict[str, Any]]]:
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            yield lineno, rec


def iter_blocks(rec: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield content blocks of a transcript record, whatever shape it has."""
    content = (rec.get("message") or {}).get("content")
    if isinstance(content, str):
        yield {"type": "text", "text": content}
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                yield block


def replay(path: pathlib.Path, prefix: str | None) -> dict[str, str]:
    """Return {file_path: final content} by applying Write/Edit in order."""
    files: dict[str, str] = {}
    for _, rec in iter_records(path):
        for block in iter_blocks(rec):
            if block.get("type") != "tool_use":
                continue
            name = block.get("name")
            inp = block.get("input") or {}
            fp = inp.get("file_path")
            if not fp or (prefix and not fp.startswith(prefix)):
                continue
            if name == "Write":
                files[fp] = inp.get("content", "")
            elif name == "Edit":
                old, new = inp.get("old_string", ""), inp.get("new_string", "")
                if fp in files and old:
                    if inp.get("replace_all"):
                        files[fp] = files[fp].replace(old, new)
                    else:
                        files[fp] = files[fp].replace(old, new, 1)
    return files


def grep(path: pathlib.Path, needle: str, context: int) -> None:
    """Print every text / tool-result payload containing needle."""
    for lineno, rec in iter_records(path):
        for block in iter_blocks(rec):
            bodies: list[tuple[str, str]] = []
            t = block.get("type")
            if t == "text":
                bodies.append(("text", block.get("text", "")))
            elif t == "tool_use":
                bodies.append(
                    ("tool_use:" + str(block.get("name")),
                     json.dumps(block.get("input", {}), ensure_ascii=False))
                )
            elif t == "tool_result":
                c = block.get("content")
                if isinstance(c, str):
                    bodies.append(("tool_result", c))
                elif isinstance(c, list):
                    for b in c:
                        if isinstance(b, dict) and b.get("type") == "text":
                            bodies.append(("tool_result", b.get("text", "")))
            for kind, body in bodies:
                if needle in body:
                    print(f"=== line {lineno} [{kind}] {len(body)} chars")
                    if context:
                        i = body.index(needle)
                        print(body[max(0, i - context):i + context])
                        print()


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("transcript", type=pathlib.Path)
    ap.add_argument("--prefix", help="only files whose path starts with this")
    ap.add_argument("--out", type=pathlib.Path, help="directory to restore into")
    ap.add_argument("--list", action="store_true", help="list recoverable files only")
    ap.add_argument("--grep", help="search text/tool-result payloads instead")
    ap.add_argument("--context", type=int, default=0, help="chars of context for --grep")
    args = ap.parse_args()

    if not args.transcript.exists():
        print(f"no such transcript: {args.transcript}", file=sys.stderr)
        return 2

    if args.grep:
        grep(args.transcript, args.grep, args.context)
        return 0

    files = replay(args.transcript, args.prefix)
    if not files:
        print("no Write/Edit calls matched", file=sys.stderr)
        return 1

    if args.list or not args.out:
        for fp in sorted(files):
            print(f"{len(files[fp]):>9}  {fp}")
        print(f"\n{len(files)} file(s). Pass --out DIR to restore.", file=sys.stderr)
        return 0

    root = args.out.resolve()
    for fp, body in sorted(files.items()):
        rel = fp.lstrip("/")
        if args.prefix:
            rel = fp[len(args.prefix):].lstrip("/") or pathlib.Path(fp).name
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(body, encoding="utf-8")
        print(f"restored {dest}")
    print(f"\n{len(files)} file(s) -> {root}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
