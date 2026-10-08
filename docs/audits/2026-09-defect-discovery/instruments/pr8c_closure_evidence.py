"""PR 8c measurement: what GitHub records about how each Phase 3 issue closed.

Plan section 8 proposition 6 reads `closedByPullRequestsReferences`. Every issue
in this run was closed by hand (merges go to `develop`, which is not the default
branch, so GitHub never auto-closes). This instrument records, per issue:

- state and stateReason;
- `closedByPullRequestsReferences` and the closer of the last CLOSED event;
- the PR numbers cited in the last comment written at or before `closedAt`,
  matched with a digit boundary so #26 never matches #263 (DC2);
- for each cited PR: state, merge commit, whether its body cites the issue with
  the same boundary, and whether the merge commit is an ancestor of the head.

Run from the repository root:

    .venv/bin/python docs/audits/2026-09-defect-discovery/instruments/pr8c_closure_evidence.py <head>
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

ISSUES = [
    258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272,
    277, 279, 281, 282, 284, 285, 286, 287, 288, 306,
]
OWNER, NAME = "nbx-liz", "LizyML"

ISSUE_QUERY = """
query($owner:String!, $name:String!, $n:Int!) {
  repository(owner:$owner, name:$name) {
    issue(number:$n) {
      state stateReason closedAt
      closedByPullRequestsReferences(first:10, includeClosedPrs:true) { nodes { number } }
      timelineItems(itemTypes:[CLOSED_EVENT], last:1) {
        nodes { ... on ClosedEvent { closer { __typename ... on PullRequest { number } } } }
      }
      comments(last:20) { nodes { createdAt body } }
    }
  }
}
"""

PR_QUERY = """
query($owner:String!, $name:String!, $n:Int!) {
  repository(owner:$owner, name:$name) {
    pullRequest(number:$n) { state baseRefName body mergeCommit { oid } }
  }
}
"""


def gh_graphql(query: str, number: int) -> dict:
    out = subprocess.run(
        ["gh", "api", "graphql", "-f", f"query={query}", "-F", f"owner={OWNER}",
         "-F", f"name={NAME}", "-F", f"n={number}"],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)["data"]["repository"]


def cites(text: str, number: int) -> bool:
    return re.search(rf"(?<![\w/])#{number}(?!\d)", text) is not None


def cited_prs(text: str) -> list[int]:
    return sorted({int(m) for m in re.findall(r"(?<![\w/])#(\d+)(?!\d)", text)})


def is_ancestor(sha: str, head: str) -> bool:
    rc = subprocess.run(["git", "merge-base", "--is-ancestor", sha, head]).returncode
    if rc not in (0, 1):
        raise SystemExit(f"git merge-base failed for {sha}")
    return rc == 0


def main(head: str) -> int:
    for n in ISSUES:
        issue = gh_graphql(ISSUE_QUERY, n)["issue"]
        closed_by = [p["number"] for p in issue["closedByPullRequestsReferences"]["nodes"]]
        events = issue["timelineItems"]["nodes"]
        closer = events[0]["closer"] if events else None
        line = (f"#{n} {issue['state']} {issue['stateReason']} closedBy={closed_by} "
                f"closer={closer}")
        print(line)
        if issue["state"] != "CLOSED":
            continue
        before = [c for c in issue["comments"]["nodes"] if c["createdAt"] <= issue["closedAt"]]
        if not before:
            print("    last comment before close: none")
            continue
        last = before[-1]
        prs = [p for p in cited_prs(last["body"]) if p != n]
        print(f"    last comment before close ({last['createdAt']}) cites {prs}")
        for p in prs:
            try:
                pr = gh_graphql(PR_QUERY, p)["pullRequest"]
            except subprocess.CalledProcessError:
                print(f"      #{p}: not a pull request")
                continue
            if pr is None:
                print(f"      #{p}: not a pull request")
                continue
            oid = (pr["mergeCommit"] or {}).get("oid")
            anc = is_ancestor(oid, head) if oid else None
            print(f"      #{p}: {pr['state']} base={pr['baseRefName']} merge={str(oid)[:7]} "
                  f"ancestor_of_head={anc} body_cites_issue={cites(pr['body'], n)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
