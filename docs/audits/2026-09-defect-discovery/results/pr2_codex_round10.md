# PR 2 — Codex review, round 10 (scope-limited, 2026-09-08)

Scoped by the rounds 8-9 relational monitor (`results/pr2_monitor_round89.md`,
**`CONVERGING` / `redirect`**) to the round-9 remedies and the two closures made
after it, under the standing constraint carried since round 9: **a finding in
existing apparatus is repaired in place or that apparatus deleted — no new
module, generator or scanner as a remedy.**

## Verdict

```
VERDICT: REQUEST_CHANGES
```

Two blocking findings at head `cbbb08d`, both reproduced here before being fixed.
**Both are overclaims**, and for both the reviewer's prescribed remedy was to
*narrow the claim*, not to add anything.

### 1 — the smart-parameter observation was not closed, and an activation could go inert (DC1, DC3, DC5, DC6)

`SMART_ACTIVATIONS` / `_names_the_resolvers_write`, and the declaration above
`SMART_PARAM_TARGETS` in production. Three faults in one construct:

- `num_leaves_ratio` was supplied **alone**, but the resolver reads it only
  inside the `auto_num_leaves` branch, so that combination was never executed —
  DC6 in the observation itself;
- the observer **overwrote** the declared `feature_weights` activation, because
  the declared value named a feature the frame does not have, so the declared
  value was never the one used;
- consequently, setting either activation to `None` left both tests green:

```
ratio activated with auto: False
stale activations: both tests PASS
undeclared write: {'num_leaves': 16, 'min_gain_to_split': 0.1} ; closure tests PASS
```

The reviewer stated the general point: *enumerating parameter names does not
enumerate accepted combinations or establish that each activation takes effect.*

### 2 — `_probe` claimed an inference its observation does not support (DC1, DC5)

Passing a control run and then reaching a substituted writer does not establish
*why* the second invocation failed: two invocations differ in the path each is
given and in whatever state the first left behind. A target that reaches the
exporter and then raises for its own reasons was reported `noticed`:

```
noticed          # the target never reads an artifact
```

---

## The remedy

**Finding 1 — supply the prerequisites, consume the declared values, and drop
the closure claim.**

`SMART_PREREQUISITES` declares what a parameter needs beneath it
(`num_leaves_ratio` needs `auto_num_leaves`), and the observation runs each
parameter with its prerequisites *and* all of them together, across the three
tasks. `feature_weights` now names a feature the frame actually has, and nothing
rewrites it. Both resolvers go through one door, so a parameter that only reaches
`resolve_ratio_params` is no longer invisible to the other. Only the
`balanced` + regression refusal is expected, and it is asserted with
`pytest.raises` rather than absorbed by a blanket handler — every other
`LizyMLError` now fails the observation.

**A new test asserts each activation changes what the resolvers produce**,
compared against the same input with its prerequisites but without the parameter
itself. That is the property both faults violated at once, and it is what stops
an activation going inert unnoticed.

**The claim is narrowed where it is made**, in the test and in the production
declaration: this is *a bounded set of executions, not a closed input domain* —
enumerating names does not enumerate combinations.

**Finding 2 — the verdict is renamed for what was observed.** `noticed` is now
`failed-after-writing`, and the docstring says plainly that this is not "it
inspected the artifact": ordering alone establishes nothing, and a target that
reaches the exporter and then fails for its own reasons lands there. What the
verdict does rule out is the shape the instrument exists for — a test that stays
green when nothing is written — and the artifact assertions in the four named
tests are reviewed directly rather than delegated to this inference.

RED verified per finding, with the reviewer's own scripts: `num_leaves_ratio` is
now activated with `auto_num_leaves`; both stale activations now fail loudly; the
injected undeclared write is caught by the table comparison; and the unrelated
post-writer failure now reports `failed-after-writing` rather than a claim about
inspection.

## Checked and clean (round 10, from the reviewer)

- **Equality implementation and ordering**: all five steps and their exception
  boundaries inspected. Identity precedes user protocols; the character
  comparison bypasses string-subclass overrides; `BaseException` remains
  uncaught.
- **Both callers**: all 32 `CASES` executed through `check_duplicate_identities`
  and `_pop_by_identity`. Both matched the table's decisions, and accepted
  adapter results retained the first object. **No wrong answer was reproduced
  beyond the documented printed-form limits.**
- **Writer discovery**: executed against an in-memory class carrying inherited,
  assigned and asynchronous export methods — all three discovered. The `export`
  prefix is an explicit bound, not an accidental substring match.
- DC2, DC4, DC7: no defect reproduced.

Its bounds, kept: head `cbbb08d` and the scoped constructs; the full suite, the
real artifact-writing tests, lint and mypy were not re-run there; closed rounds
and persistence correctness not reopened; and finding 1's injected write
demonstrates undetected future drift, not an existing undeclared production
write.

## State handed to the maintainer

Blocking per round: **1, 1, 2, 2, 1, 3, 2, 4, 3, 2**. Twenty-one findings, every
one reproduced and every one fixed.

**No finding in this round was a production defect.** Both were claims the
apparatus made about itself that exceeded what it does — and for the first time
the whole remedy was subtraction plus one property: nothing was added but the
prerequisite declaration and the effect assertion, and two declarations were cut
back to what they deliver.

Full suite **2409 passed**; `ruff check .`, `ruff format --check .`,
`mypy lizyml/` clean. The rounds 9-10 relational monitor runs before round 11.
