# LizyML defect-discovery plan — state

- Head under discovery: `5712f41`. Working tree clean throughout.
- Discovery plan: `discovery-plan.md` v12, APPROVED at review round 11.
- **Repair plan: `phase3-plan.md` revision 8. Codex round 7: zero blocking
  findings, Change Gate compliant. Not yet formally APPROVED** — round 7
  returned `REQUEST_CHANGES` on two stale numbers, both now fixed, and round 7
  had been declared final to the reviewer so no round 8 was run. **Resume with
  one confirmation round if the APPROVE stamp is wanted.**
- `/tmp` survives the session, **not a reboot — and it was lost once.** Recover
  by replaying `Write`/`Edit` tool_use blocks from the session transcript at
  `~/.claude/projects/-home-rem-repos-LizyML/<session>.jsonl`; that recovered
  all 68 files. The GitHub issues are the only durable record.

## Regenerating every measured figure

`bash run-exclusive.sh regen bash regen.sh` — runs all measurements, the
manifest derivations, the completion-tool tests, the revision validation, and
the deliverables' RED baseline. **Never transcribe a number the plan quotes;
read it out of this.** Two rounds found stale figures that had been copied by
hand.

Run everything with `/home/rem/repos/LizyML/.venv/bin/python` from
`/tmp/lizyml-discovery-plan`. **Do not use `uv run`** (read-only uv cache).

## Phase status — PHASE 1 AND 2 COMPLETE, AUDITED, REMEDIATED

An independent execution audit (`reviews/exec-r1.json`, 23 counts recomputed)
returned `REQUEST_CHANGES` on the first pass with 8 blocking and 3 major
findings, and was right on every substantive point. All are now closed.

| | first pass | after remediation |
|---|---|---|
| members the declared procedure decided | 2,655 / 3,260 = 81.4% | **3,286 / 3,286 = 100%** |
| members undecided | 608 | **0** |
| steps meeting all three exit conditions | 0 / 8 | **8 / 8** |

`gap.py` counts this from the recorded result files. An earlier version of it
hard-coded several steps complete and printed 100% — the stale-acceptance shape
the plan refuses — and was rewritten to derive everything.

| Step | Population | State |
|---|---|---|
| D1a | 245 | 219 executed cells + R7's 26 boundary cells (`lgb.Dataset` kwargs, 9 generated-code sites), 0 UNKNOWN |
| D1b | 20 | gate reads `ast.Raise` subtrees only, executed against head |
| D2 | 162 | attribution fixed; 15 misattributed `AGREE` rows became `CANNOT-TELL` |
| D3 | 527 | sets 1–4 redone against the declared joins; the contract check settled the last 5 |
| D3b | 44 | all executed as real tuning jobs; control observed, not inferred |
| D4 | 351 | every stage driven by a config matrix; `UNCLASSIFIED` 60 → 0 |
| D5 | 134 | every non-`OK` row filed or disposed |
| D6 | 1803 | 184 candidates settled: 179 confirmed hollow, 5 downgraded |

**Filed:** #261–#272, plus comments on #258, #263 and #270. Register:
`results/FINDINGS.md` (its final section records what the audit changed).

## Running anything heavy

```
bash run-exclusive.sh <label> <command...>
```

`flock`-based, one job at a time. LightGBM uses every core per `lgb.train`, and
`ps` in this sandbox shows only the current shell — a job can look dead while it
is very much alive, so check by `TaskStop`ping it, not by `ps`. This wrapper
exists because "one job at a time" was written down after the first incident
(load 90) and broken anyway on the second (load 63.5).

## Reproducing the trace

`d6.py` and D3 set 5 both read the output of one instrumented pytest run:

```
cd /home/rem/repos/LizyML
PYTHONPATH=/tmp/lizyml-discovery-plan/instruments \
  .venv/bin/python -m pytest tests/ -p no:cacheprovider --no-cov -q -p trace_plugin
```

It writes `results/d6_traces.jsonl` and `results/d3set5_bindings.json`.
**Takes ~260 s on an idle box.** Two gotchas that cost hours:

1. `trace_plugin` opens the trace file with `"w"` **at import**, so *any* pytest
   run with `-p trace_plugin` — including a one-file smoke test — truncates it.
   Never start a second one while the first is running, and copy the outputs to
   `*.full.*` the moment a run completes.
2. LightGBM uses every core per `lgb.train`. Two traced suites plus a D4 run put
   this 32-core host at load 90 and slowed the suite from 260 s to 22 minutes for
   24 tests. `ps` inside this sandbox shows only the current shell, so a
   background job can look dead while it is very much alive — check by
   `TaskStop`ping it, not by `ps`. Run one CPU-heavy job at a time.

`results/d3set5_bindings.partial-1585.json` is a preserved partial (~1585 of
2051 items) carrying the `n_prod` caller-origin split; it produced 5 of 6 D3
controls and is the fallback if a full run cannot be had.

## Files

| Path | What |
|---|---|
| `d2.py` / `d2_classify.py` / `d2_judgements.py` / `d2_final.py` | D2, in that order |
| `d3.py`, `d4.py`, `d5.py`, `d6.py` | one step each |
| `instruments/trace_plugin.py` | the pytest plugin feeding D3 set 5 and D6 |
| `results/FINDINGS.md` + `results/FINDINGS_D5.md` | the findings register (fold D5 in) |
| `results/d*_rows.jsonl`, `results/d2_final.jsonl` | one row per member, with verdict and evidence |
| `patches/` | the D1a and D1b deliverable tests, both failing on head by design |
| `summarize.py` | `python summarize.py d5_rows.jsonl` — verdict tallies |

## Filed so far

#261 `feature_weights` inert · #262 tuning `category: model` unvalidated ·
#263 three unraised `ErrorCode` members · #264 `fit(params=)` not forwarded ·
comment on #258 (10 of 22 metric entries).

## Next: Phase 3 (repair)

Nothing in this working set is a Proposal and none of it grants implementation
authority. The repair order per confirmed defect is `discovery-plan.md` §5:

1. decide the specification (BLUEPRINT / HISTORY Proposal, CLAUDE.md §2),
2. bring the implementation to the decided specification,
3. add the regression test pinning the observed artifact — and where the defect
   belongs to a population, extend that population's permanent test rather than
   adding a single case.

**#265 comes first**: two other dispositions wait on which BLUEPRINT clause is
authoritative. **#270 should land with or before #261 / #262 / #264** — those are
the gates that let them through, and closing a defect while its gate still
asserts one step short leaves the class open.

The two tests in `patches/` fail on head by design and are the step-3 artifacts
for D1a and D1b.

Issue style follows `lizy-harness` `docs/deepseek-harness-review.md`: stable
finding ids, grouped by mechanism, a disposition table naming the carrying
artifact, no invented Issue numbers. Draft bodies in English from the first
draft (`validate-pr-language.sh` blocks Japanese in a `gh` command) and file with
`gh issue create --body-file`.
