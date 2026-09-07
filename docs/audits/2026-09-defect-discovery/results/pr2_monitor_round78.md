# PR 2 — relational monitor, rounds 7-8 (2026-09-08)

Spawned before round 9, per `policy:loop-monitor`. Given round 8's four findings
unsoftened, including that **two of them were inside the two instruments its own
predecessor prescribed**, and asked one further question: after round 8's fixes,
is any declaration in scope still verified by a hand-written table rather than by
a construction or by a stated limit?

```
VERDICT: DRIFTING
RECOMMENDATION: redirect
```

The second `DRIFTING` of this run, and the first that came with a measured
denominator.

## What it measured

> *Forwarding path* — the declared deliverable (`model.py`, `adapter.py`,
> `provider.py`, `smart_params.py`, `param_names.py`, `_model_factories.py`):
> last touched at `9b53aa5`, **before round 6 was reviewed**. Rounds 6, 7 and 8
> changed zero lines of it.
>
> *Apparatus*: +637 test lines and +379 doc lines in one round.
> `test_value_equality.py` plus the override instruments now run **2144 lines
> against 146 lines of helper**.
>
> Apparatus share of findings: 0/2 in round 7 → **2/4 in round 8**.

## It also recorded the one contraction

> `value_equality.py` churned 109 lines R7→R8 for a net +3, and it **lost a
> step**. Round 8's largest remedy was a deletion, not another guard — the
> loop's first contraction and a change in kind: the class "a guard hypothesises
> what iterating a comparison yields" can no longer recur.

That is held alongside the verdict, not against it: *the deliverable has been
clean and unchanged for three rounds, every round-7/8 finding sits in the helper
or its apparatus, and this round two findings existed only because the previous
round's remedy created them. That is the drift shape, and it holds despite the
contraction.*

## The table question — three answers, all reproduced here before acting

**1. Round 8's own cost statement was executed by nothing.** The docstring says
dtype-differing arrays are refused and identically-printing arrays are reported
the same. No case in the file touched dtype or `repr` summarisation. The monitor
ran it rather than reading it; so did this context:

```
values_differ(np.array([1, 2]), np.array([1.0, 2.0]))          -> True
two 2000-element arrays differing at index 1000, reprs equal   -> False
```

Measured true, pinned by nothing. **Fixed**: both halves are now cases in the
table, with a separate test asserting the premise — that `repr` really does
summarise the differing element away — so the case cannot pass for the wrong
reason if numpy changes its threshold.

**2. `_ARTIFACT_WRITERS` bounded the exporting population by hand.** Complete
today, because `_model_persistence.py` defines exactly `export` and
`export_code`, but nothing tied the constant to that module: a third writer would
drop its tests from the population in silence. **Fixed**: the set is read from
the class that defines the writers, `_probe` substitutes every member of it
through an `ExitStack`, and a test pins that the derivation still finds the two
that exist.

**3. `test_equal_values_of_any_shape_are_accepted_under_two_spellings` bound one
object under both keys**, so all six cases were answered by the identity step and
reached nothing else. The claim "equal values under two spellings are accepted
whatever the type" was carried by six comparisons that never compared.
**Fixed**: the table holds factories, the test builds two objects and asserts
they are distinct, and the two genuine singletons (`None`, `True`) are named
rather than skipped.

Fixing it found a fourth instance immediately: `lambda: (1.0, 2.0)` returned
**the same tuple twice**, because CPython folds a constant tuple into the code
object. The distinctness assertion caught it on its first run — the tuple case
had never compared anything either.

## Its recommendation, adopted

> **`redirect`** — round 9 may fix, but **may not ship a new test module,
> generator or scanner as a remedy** (a finding in existing apparatus is repaired
> in place, or that apparatus is deleted). This is the inverse of monitor 67's
> redirect, whose added instruments produced two of round 8's four findings, and
> it corrects the denominator behind the maintainer's premise rather than
> overturning it.

Adopted in full. Every repair above is in place in an existing module: three
cases and one premise test added to an existing table, one constant derived
instead of written, one fixture turned from values into factories. No new module,
generator or scanner.

Full suite **2409 passed**.
