# PR 0 — Codex review, round 3 (2026-09-07)

Rounds 1 and 2 are in the sibling files. A relational loop monitor observing
rounds 1–2 ran in a fresh context before this round's prompt and returned
`CONVERGING` with recommendation `continue`; it also corrected the capsule's
framing, establishing that `test_declared_versions.py` is a declared component of
PR 0 rather than apparatus grown by the loop.

---

## Verdict

```
VERDICT: REQUEST_CHANGES
```

### Findings

- **blocking** — `tests/test_docs/test_declared_versions.py:64`: the declaration
  grammar remains open. Importing the test module and classifying
  counter-examples produced:

  ```text
  format_version=2-bogus     => [('format_version', '2', 'value')]
  config_version: int = 999  => [('config_version', 'int', 'annotation')]
  ```

  Punctuation-suffixed junk is accepted as the current integer, while a plausible
  typed declaration silently ignores its initialiser. Both violate H-0092's
  exhaustive accept/annotation/reject claim (DC1/DC2). Round 2's exact
  `format_version=2bogus` example is resolved, but the repair opened equivalent
  boundary holes.

- **blocking** — `phase3-plan.md:1195`: §4 records the deferral, but authoritative
  §8 still states that the completion tool and manifest "both run", that 29 tests
  ship and pass, and that derivations report 15/15. This contradicts
  `instruments/deferred/README.md`, which reports 14/15 and says the artifacts do
  not work. `pytest --collect-only docs/audits/2026-09-defect-discovery` collects
  only **21 tests**, and the README's stale-items list does not disclose that
  eight of the promised 29 unit tests are missing. The record retains false
  completion claims and understates the uncovered guarantee — DC5.

### Checked and clean

- The three targeted test files passed: **40 passed, 1 warning**.
- The round-1 regression fallback repair remains correct; BLUEPRINT §§10.3.3 and
  10.6.2 match the implementation.
- Automatic inner-valid resolution inherits the outer gap; explicit
  `time_holdout` retains `gap=0`.
- The four manifest defects the README documents were confirmed independently:
  #271 derives 93 against a declared 92; #265 and #266 name nonexistent nodes;
  future-PR nodes are intentionally absent.
- `ruff check --no-cache .` passed; linting the archived instruments explicitly
  produced errors only under `docs/audits/`.
- No package or active test imports the archive; pytest discovery is restricted
  to `tests`.
- `git diff --check` and `ruff format --check .` passed.
- H-0092's scope matches the actual production, documentation, test, Ruff and
  archive changes.
- No additional DC3, DC4, DC6 or DC7 defect was found.

---

## Disposition (main context)

Both findings upheld.

### Finding 1 — the grammar was still token-shaped

The round-2 repair matched a *token* (`[\w.]+`) and classified it. Any token
pattern stops somewhere, and what follows the stop is invisible: `2-bogus` stops
at the hyphen, and `int = 999` stops at the space. The fix is structural — match
only the **prefix** (name and separator) and classify the **whole remainder of
the line**:

| Outcome | Rule |
|---|---|
| value site | digits whose *next* character is end-of-line, whitespace, or an enumerated closing mark (`"` `` ` `` `)` `]` `}` `>` `,` `;` `）` `」` `』` `。`) |
| type annotation | the **entire** remainder is one admitted type name |
| unreadable | everything else — reported as a failure |

Matching the remainder also fixes a hole neither round found: two declarations on
one line each get their own remainder instead of the second being swallowed.

Verified against every counter-example the gate has produced, plus the seven real
sites:

```
REAL      format_version=2"]  / +FORMAT_VERSION = 2 / +config_version: int
          format_version=2）  / `config_version=1` で… / `FORMAT_VERSION = 2`（…
          -> accepted (6 value sites, 1 type annotation)

HOSTILE   2bogus / 2-bogus / int = 999 / 2.5 / (empty) / two / 02x
          -> all REJECTED
          format_version=1 and format_version=2bogus
          -> site value=1  AND  REJECTED '2bogus'   (both classified)

NOT A DECLARATION  `format_version` を必須にする。 /
                   **format_version rejection**: … / validate format_version
          -> no match
```

`tests/test_docs`: **13 passed**. H-0092's acceptance criteria now state the
remainder rule and name the counter-examples.

### Finding 2 — the record contradicted itself

§8 is the authoritative statement of how Phase 3 completion is measured, and it
narrated an instrument that no longer exists in the state described. A deferral
recorded in §4 while §8 still claims the tool runs is the DC5 shape with the
sections swapped.

- §8 now opens with a superseding banner naming each false sentence — "both exist
  and both run", "29 tests, passing", "15 ok, 0 disagreeing" — and stating that
  the section is kept as the specification a repair must satisfy, not as a report
  of what runs.
- `instruments/deferred/README.md` gains the fifth defect: `test_phase3_gap.py`
  collects **21**, not 29, and **nothing records which eight propositions the
  missing tests covered** — so the instrument's own coverage is unknown, not just
  unshipped. It also states explicitly that §8's narration is false of everything
  in the repository and that this README supersedes it.

Gates after both repairs: `ruff check .`, `ruff format --check .`,
`mypy lizyml/` clean; full suite **2069 passed**.
