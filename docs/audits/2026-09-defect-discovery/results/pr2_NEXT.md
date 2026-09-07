# PR 2 — where this stands, and the next action

Written as a clean stopping point. Everything below is committed and pushed;
the working tree is clean and CI is green on the branch.

## State

- PR **#278**, draft, branch `fix/phase3-pr2-fit-params-forwarding`, base
  `origin/develop` = `ccae32b`.
- Full suite **2383 passed**; `ruff check .`, `ruff format --check .`,
  `mypy lizyml/` clean.
- **Seven review rounds, no `APPROVE`.** Blocking per round: 1, 1, 2, 2, 1, 3,
  2 — twelve findings, every one reproduced and every one fixed.
- Records: `pr2_codex_round[1-7].md`, monitors `pr2_monitor_round[12,23,34,45,56,67].md`,
  decision D7 in `DECISIONS-PENDING.md`.

## The maintainer's standing instruction

**Run until the external reviewer returns `APPROVE`**, on the standard applied
to PR 1 (which took six rounds and got there on a scope-limited sixth). Stated
reasoning: not obtaining `APPROVE` is itself evidence that real problems remain
in the fix code — which the record supports, since every round found real,
reproduced defects.

## What was done before round 8, and why it should be different

The rounds 6-7 monitor identified why the previous three rounds each found
something: **each remedy shipped a declaration verified by a hand-written table,
and the next round found the gap between declaration and verification.** On that
method, "run until APPROVE" manufactures its own next finding.

Three things changed, all committed:

1. **One more self-authored defect found and fixed** — round 7's widening moved
   an unbooleanable comparison into the elementwise reduction, where iterating
   yielded truthy junk and two different values read as equal. The string
   special-case is replaced by the property it stood for: an element with no
   `__bool__` of its own is not a comparison outcome.
2. **Instrument 1** — the no-raise bound quantified over a generated cross
   product of the dunders `values_differ` touches, in both directions, with the
   population derived so a shrunk behaviour table fails.
3. **Instrument 2** — every artifact-reading test must fail when nothing is
   written, over a population found by reading the module. **It found a second
   instance of round 7's defect on its first run** (the `export_code` test
   passed under a no-op `export`, which never touched the writer it uses).

## The next action

**Run round 8**, scope-limited, using
`scratchpad/codex-pr2-review-prompt-r7.md` as the shape. Per the rounds 6-7
monitor, point it at:

- the round-7 remedies (the handler widening, the artifact-reading test),
- the two instruments above,
- **rounds 1-6 surfaces explicitly out of scope, and no new region admitted.**

Record in the prompt that the persistence/export redirect already measured the
last unmeasured shipped surface and found production correct, so it is not fresh
territory. The prompt's metadata block takes `Monitor-mode: relational`,
`Monitor-verdict: CONVERGING`, `Monitor-evidence: results/pr2_monitor_round67.md`,
`Monitor-disposition: redirect`.

**Do not pre-register this as the last round.** The maintainer's standard is
`APPROVE`, and two earlier pre-registrations were overridden.

## The mechanics, unchanged

```
CODEX_HOME=<scratchpad>/codex-home codex exec --sandbox read-only \
  -C /home/rem/repos/LizyML --color never - < <prompt> > <log> 2>&1
```
Copy `auth.json` + `config.toml` in with `setup_codex_home.py`, and **delete the
copy afterwards** with `cleanup_codex_home.py`. Before every round from 3
onward, a relational monitor runs first; the hook validates the metadata block.
