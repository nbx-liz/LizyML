# PR 2 — Codex review, round 7 (redirected, 2026-09-07)

Rounds 1-6 are in the sibling files. The rounds 5-6 monitor returned
**`DRIFTING`** — the first non-converging verdict in this run
(`results/pr2_monitor_round56.md`) — and pointed the round away from the value
helper and at the one shipped surface six rounds never touched: persistence and
export of an overridden fit.

---

## Verdict

```
VERDICT: REQUEST_CHANGES
```

Two blocking findings. **Both are in code written in rounds 5-7**, which is the
pre-registered stop condition, and the reviewer closed the loop itself:

> Both findings concern rounds 5–7, so the stated stop condition closes this
> review loop; this verdict does not initiate round 8.

### 1 — the declared exception bound had an unguarded path (DC5)

`lizyml/core/value_equality.py`. The truth step caught `ValueError` and
`TypeError` — the two an array raises — so a comparison object whose `__bool__`
fails for a reason of its own escaped:

```python
class Comparison:
    def __bool__(self): raise RuntimeError("truth failed")
class Value:
    def __eq__(self, other): return Comparison()

values_differ(Value(), Value())   # -> RuntimeError: truth failed
```

So "the function no longer raises for any input", written into the monitor
record before this round, exceeded the implementation. The reviewer also
attributed it precisely — `git blame` on that handler — rather than asserting it.

### 2 — the exported-booster test passed without exporting anything (DC1)

`tests/test_core/test_fit_params_override.py`. It asserted on
`_booster_text(model)`, the **in-memory** model, so with `Model.export` replaced
by a no-op it still passed:

```
PASSED; export calls: 1
```

A test written in the same pass that was supposed to close an unexamined axis,
and it was green because it did not look at what was written. The exact class
this PR has been fixing, in a test about it.

---

## The remedy

**Finding 1** — the truth step now catches every exception, on the same
reasoning the rest of the function already used: the comparison result is an
object the *caller's value* produced, so it can fail for any reason, and a
function declared not to raise cannot enumerate which. Added to the case table
as "a comparison whose truth value raises".

**Finding 2** — the test now loads the artifact and asserts the override on
**both** persisted surfaces: every CV booster and the refit model `predict`
uses. And the property the reviewer used to expose it is pinned, so the same
substitution cannot pass again:

```python
assert restored is not model and restored.fit_result is not model.fit_result
```

RED verified: narrowing the truth handler again reddens three cells; asserting
on the in-memory model again reddens the export test; and with `export` patched
to a no-op the test now fails on the missing artifact instead of passing.

---

## Checked and clean (round 7)

- 61 passed on the value-equality file — and the reviewer said plainly that
  "these passing cases do not establish the claimed bound", which is why the
  declaration was the finding rather than the cases.
- **Persistence behaviour**, exercised through the real exporter, loader and
  joblib with an in-memory filesystem: loaded predictions matched exactly,
  evaluation matched, the saved refit booster kept `0.5`, metadata kept the
  config's `0.001` and a tuned `0.07`, subsequent fits used `0.001` / `0.07`,
  and a new override used `0.3`. Both an untuned and a constructed
  `TuningResult` case. Stated bound: this validates serialisation and the
  restored overlay, **not disk I/O and not a real tuning run**.
- Export params: 9 passed, 9 setup errors from the sandbox having no writable
  temporary directory — reported as environmental, not as findings.
- DC1–DC7 considered; no additional reproduced finding. `Model.simulate` does
  not exist, and its absence was correctly not treated as an in-scope defect.
- No files changed.

## State handed to the maintainer

Blocking findings per round: **1, 1, 2, 2, 1, 3, 2**. Twelve findings, every one
reproduced and every one fixed.

**The stop condition tripped, and both the run and the reviewer stopped on it.**
Round 8 is not this run's to open.

The trajectory the rounds 5-6 monitor named is now four rounds long: **no round
since round 4 has found a defect that predates this PR.** Rounds 5, 6 and 7
found defects in code rounds 4, 5 and 6 wrote — and round 7's second finding was
in a test written in the same pass as the round it was meant to close.

Full suite **2349 passed**; `ruff check .`, `ruff format --check .`,
`mypy lizyml/` clean. The decision is **D7**, updated with this round.
