# PR 0 — Codex review, round 4 (2026-09-07) — APPROVE

Rounds 1–3 are in the sibling files. This round was **redirected** rather than
run on the same shape as its predecessors, for the reason below.

---

## The redirect

A relational loop monitor observing rounds 2–3 ran in a fresh context and
returned **`VERDICT: DRIFTING`** with recommendation `take-stop-condition`. The
signal was one the previous monitor had named *in advance*: a round-3 blocking
finding again confined to `test_declared_versions.py` would be the second
consecutive round matching the drift shape.

Its discriminator was not location but input population:

> The parser is a declared PR 0 component, so a finding on it is not drift by
> location. The check's production inputs are the seven real sites, correctly
> classified since round 1 and confirmed accepting in round 3's own clean list.
> Rounds 2 and 3 hardened against strings present in **no document in the
> population**. The deliverable was declared clean and did not change for two
> consecutive rounds while the periphery deepened. `VALUE`'s closing-mark
> lookahead is still an enumerated list, so a round-4 parser finding is
> predictable from that axis — but it would now surface as a **loud failure on a
> novel closing mark, not a false-clean**. Further rounds there trade only
> false-fail risk, which is the safe side to stop on.

**Main-context disposition: `redirect`.** The finding was accepted in full and
the recommendation adopted as to the *axis*, not the gate — the run's merge
condition is an explicit Codex APPROVE, which the main context does not waive on
its own judgement. So round 4 ran with:

- the DRIFTING verdict and its reasoning disclosed to the reviewer up front;
- the objective narrowed to whether round 3's two findings were closed and
  whether the deliverable — untouched since before round 2 — was still correct;
- **further hardening of the value grammar against inputs absent from the
  document population declared out of scope**, with one exception preserved: a
  synthetic string was still admissible as a finding if it produced a
  *false-clean*. A string that merely fails loudly was not.

The reviewer was invited to object to the scoping in one sentence and proceed. It
did not object.

---

## Verdict

```
VERDICT: APPROVE

## Findings

None.
```

### Checked and clean

- Round-3 grammar finding is closed. Direct classification rejected both
  `format_version=2-bogus` and `config_version: int = 999`. The scan found all
  seven real value sites with correct values, one valid type annotation, and
  **zero unreadable declarations**.
- Round-3 record finding is closed. `phase3-plan.md` §8 clearly supersedes the
  obsolete completion claims; `instruments/deferred/README.md` identifies the
  unshipped instrument, stale components, the 21-versus-29 test discrepancy, and
  the uncovered Phase 3 completion guarantee.
- Focused tests passed: **40 passed, 1 warning** across the three requested test
  files.
- `BLUEPRINT.md` §§8.2, 10.3.1, 10.3.3 and 10.6.2 match `inner_valid.py` and
  `_model_factories.py`, including the regression fallback and the
  automatic-versus-explicit gap behaviour.
- All three `ARCHITECTURE.md` declarations state `format_version=2`, matching
  `persistence/exporter.py`.
- H-0092's scope matches `git diff origin/develop` plus the untracked audit and
  document-test trees.
- `ruff check --no-cache .` and `git diff --check` passed.
- `docs/audits/**` remains excluded from Ruff, is outside default pytest
  discovery via `testpaths = ["tests"]`, and is not imported by `lizyml/` or by
  active tests.

---

## What the four rounds cost and returned

| Round | Verdict | Findings | Where they landed |
|---|---|---|---|
| 1 | REQUEST_CHANGES | 1 blocking, 1 major | production source + the check's DC1 guard |
| 2 | REQUEST_CHANGES | 1 blocking | the check's value grammar (DC2) |
| 3 | REQUEST_CHANGES | 2 blocking | the value grammar again, and the run's own record (DC5) |
| 4 | **APPROVE** | none | — |

Round 1 is the one that paid for the loop: it found a real behaviour defect
(`BlockedGroupInnerValid` returning an empty inner-train for regression) that no
document change could have fixed, because the implementation was the wrong side
of the contradiction. Rounds 2 and 3 hardened the permanent check and corrected
the record. The monitor's judgement that rounds 2–3 were periphery is recorded
here as accepted, not disputed: the redirect is what kept round 4 from being a
fourth pass over the same axis.

Gates at APPROVE: `ruff check .`, `ruff format --check .`, `mypy lizyml/` clean;
full suite **2069 passed**.
