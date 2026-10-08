"""PR 8c measurement after design review round 1.

Two questions the round 1 findings raise, measured on the real records:

1. Closure association (finding 2). For each closed row and each fixing PR:
   - does the PR body cite the issue after a keyword (fix/fixes/fixed, close/closes/
     closed, resolve/resolves/resolved, ref/refs), digit-bounded?
   - does the last issue comment at or before closedAt name the PR, either as a
     digit-bounded `#N` or by the PR's exact title (a trailing " (#N)" removed)?
2. Executed outcomes after the fix (finding 3). For each row, run its tests at the
   current tree with a JUnit report and count passed / skipped / xfailed / failed
   per test file, so the allowed non-pass outcomes can be declared from data.

    .venv/bin/python .../pr8c_round1_probe.py <rows.json> <fixing-prs.json>
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

OWNER, NAME = "nbx-liz", "LizyML"
PY = str(pathlib.Path(".venv/bin/python").absolute())
KEYWORD = r"(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?|refs?)"

Q = """
query($owner:String!, $name:String!, $n:Int!) {
  repository(owner:$owner, name:$name) {
    issueOrPullRequest(number:$n) {
      __typename
      ... on Issue { state closedAt comments(last:30) { nodes { createdAt body } } }
      ... on PullRequest { title body }
    }
  }
}
"""


def gh(n: int) -> dict:
    out = subprocess.run(
        ["gh", "api", "graphql", "-f", f"query={Q}", "-F", f"owner={OWNER}",
         "-F", f"name={NAME}", "-F", f"n={n}"],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out)["data"]["repository"]["issueOrPullRequest"]


def closure(fixing: dict[str, list[int]]) -> None:
    print("== closure association")
    for issue, prs in sorted(fixing.items(), key=lambda kv: int(kv[0])):
        data = gh(int(issue))
        if data["state"] != "CLOSED":
            print(f"#{issue}: {data['state']} (not judged)")
            continue
        before = [c for c in data["comments"]["nodes"] if c["createdAt"] <= data["closedAt"]]
        last = before[-1]["body"] if before else ""
        for pr in prs:
            p = gh(pr)
            body_kw = re.search(rf"\b{KEYWORD}\s+#{issue}(?!\d)", p["body"], re.I) is not None
            title = re.sub(r"\s+\(#\d+\)$", "", p["title"])
            by_number = re.search(rf"(?<![\w/])#{pr}(?!\d)", last) is not None
            by_title = title in last
            print(f"#{issue} <- #{pr}: body_keyword_cites={body_kw} "
                  f"close_comment_by_number={by_number} by_title={by_title}")


def outcomes(rows: dict) -> None:
    print("== executed outcomes at the current tree")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for issue, row in sorted(rows.items(), key=lambda kv: int(kv[0])):
        with tempfile.TemporaryDirectory() as d:
            xml = pathlib.Path(d) / "r.xml"
            subprocess.run([PY, "-m", "pytest", *row["tests"], "-q", "--no-cov", "-p",
                            "no:cacheprovider", f"--junitxml={xml}"],
                           capture_output=True, text=True, env=env)
            counts: dict[str, int] = {}
            for case in ET.parse(xml).getroot().iter("testcase"):
                kids = [k.tag for k in case]
                if "failure" in kids or "error" in kids:
                    kind = "failed"
                elif "skipped" in kids:
                    msg = next(k for k in case if k.tag == "skipped").get("type", "")
                    kind = "xfailed" if "xfail" in msg else "skipped"
                else:
                    kind = "passed"
                counts[kind] = counts.get(kind, 0) + 1
        print(f"#{issue}: {dict(sorted(counts.items()))}")


def main(rows_path: str, fixing_path: str) -> int:
    rows = json.loads(pathlib.Path(rows_path).read_text(encoding="utf-8"))
    fixing = json.loads(pathlib.Path(fixing_path).read_text(encoding="utf-8"))
    closure(fixing)
    outcomes(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
