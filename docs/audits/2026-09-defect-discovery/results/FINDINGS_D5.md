## D5 — residual closed-set sweeps

**Population:** 134 members in four sets, each count reproduced against the
plan — 40 `except` clauses (39 from an AST sweep of `lizyml/` plus 1 inside a
generated-code template), 13 `warnings.warn` sites, 57 `H-00xx` proposals absent
from `BLUEPRINT.md`, and 8 concrete splitters × 3 roles.

**Result:** 134 classified, **0 unclassified**. 64 `OK`, 56 `DEFECT`,
14 `CANNOT-TELL`. Stop condition **REACHED**.

**Positive control re-detected:** H-0083 — the SHA-256 artifact checksum was
decided, is implemented (19 of its 21 named identifiers exist in `lizyml/`), and
`BLUEPRINT.md` contains the word "checksum" **zero** times.

**An instrument defect found here changed two steps.** `HISTORY.md` delimits
proposals two ways: older ones open with `- ID: H-00xx`, newer ones with a
`## H-00xx:` heading. The first rule matched only the new form, the second only
the old, and each silently hid half the register — including this step's own
control. Both forms are now accepted. **The same wrong rule was in D2**, where it
tagged every line `None` and dropped `HISTORY.md` from every comparison while the
retrieval reported it as searched. D2 was re-run after the fix.

A second one, in the same sweep: `except ImportError: pass` at ten sites was
first reported as a silent swallow. It is the optional-dependency guard — a
module-scope import with a sentinel that every user of the module checks, which
is the written policy in `.claude/skills/optional-dependencies/SKILL.md`. Ten
false positives out of thirteen.

### D5-F1 — a leakage validator skips the columns it cannot compare

`lizyml/data/validators.py:94`:

```python
except (TypeError, ValueError):
    # Non-comparable types; skip
    pass
```

The leakage check walks candidate columns and, on a column whose comparison
raises, skips it silently. The validator then returns its warning list, and a
column that was never examined is indistinguishable from a column that was
examined and found clean.

This is the canonical DC1 wording from the defect-class register — *"couldn't
match, so skip"* — inside a validator whose whole job is to detect leakage. The
one place in the library that must not report "clean" when it means "did not
look".

**Recommended:** count the skipped columns and surface them, or fail closed.
Either way the caller must be able to tell "no leakage found" from "n columns
were not checked".

### D5-F2 — 53 of 57 proposals decide contract-surface matters and never reach BLUEPRINT

`HISTORY.md` carries 92 proposals; `BLUEPRINT.md` names 35 of them. Of the 57
absent, 53 decide something on the surface `CLAUDE.md` section 3 requires
BLUEPRINT to fix (contract types, `format_version`, split/leakage/calibration
boundaries, migration), and every one of the 53 has its named identifiers present
in `lizyml/` — the DC3 shape: decided, implemented, spec never updated. **None**
is DC5; no proposal in the set was decided and left unimplemented.

Whether each individual proposal owed BLUEPRINT an update is a maintainer
judgement, so the measured, non-judgemental facts are these: 57 of 92 proposal
ids are absent from BLUEPRINT, and for H-0083 the word "checksum" appears in
BLUEPRINT zero times while the mechanism is load-bearing in the Artifacts
contract that `CLAUDE.md` section 3 requires BLUEPRINT to fix.

**Recommended:** decide the 53 in one pass, and add a check that a proposal
touching the contract surface cannot be marked done while BLUEPRINT does not name
it. That check is the durable repair; folding 53 entries in by hand is not.

### D5-F3 — an unreadable data file yields a fingerprint of `None` that nothing checks

`lizyml/data/fingerprint.py:50` returns `None` when the file cannot be read, and
its docstring says so — a written policy, so the handler itself is `OK`. What
makes it a finding is the consumer: issue #263 established that
`ErrorCode.DATA_FINGERPRINT_MISMATCH` is raised by nothing, so the recorded
fingerprint is never compared against anything.

Composed, the two produce a path where an unreadable source file gives a null
fingerprint that no code inspects. Neither half is wrong alone; the pair is.

**Recommended:** fix as part of #263 — whatever code learns to verify the
fingerprint must treat `None` as a failure to verify, never as a match.

### D5-F4 — the inner role cannot be observed for any splitter

Eight of the 24 splitter × role cells are `CANNOT-TELL`, all of them the inner
role: inner validation is served by `lizyml/training/inner_valid.py` strategies,
not by splitters, so a splitter never occupies that role and its time/group
constraint cannot be observed there. Recorded rather than passed as `OK` — a
constraint never checked in a role must not be reported as honoured in it. The
calibration role reuses the outer splits verbatim (H-0058, `BLUEPRINT.md:571`),
so the outer verdict carries over and those 8 cells are `OK`.

### D5 dispositions

| # | Action | Carrying artifact |
|---|---|---|
| 1 | Make the leakage validator report skipped columns instead of dropping them (D5-F1) | New Issue |
| 2 | Decide the 53 unfolded proposals; add a proposal-vs-BLUEPRINT drift check (D5-F2) | New Issue; H-0083 is the canonical case, shared with D2-F5 |
| 3 | Treat a `None` fingerprint as failure-to-verify when #263's check is implemented (D5-F3) | Comment on #263, not a separate Issue |
| 4 | None — the inner-role cells are recorded, not a defect (D5-F4) | This register |
