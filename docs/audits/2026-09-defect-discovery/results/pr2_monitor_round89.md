# PR 2 — relational monitor, rounds 8-9 (2026-09-08)

Spawned before round 10. Given the numbers unsoftened — blocking per round
**1, 1, 2, 2, 1, 3, 2, 4, 3**, round 9's apparatus share **2 of 3**, and both of
those in the same two constructs round 8 had found — and asked one further
question: **does anything left in scope make a claim over an open population?**

```
VERDICT: CONVERGING
RECOMMENDATION: redirect
```

The first `CONVERGING` since rounds 6-7, and the first reached over a measured
contraction rather than over an argument.

## What it measured

> *Deliverable — unchanged.* `model.py`, `adapter.py`, `provider.py`,
> `smart_params.py` were last touched at or before `9b53aa5`, before round 6 was
> reviewed. Round 9 changed only `value_equality.py`, **+12/−1**.
>
> *Periphery — contracted, for the first time in the run.* 2144 → **2040** test
> lines against 146 → 159 of helper; net **−64** after **+431** in round 8.
> Findings fell 4 → 3.

Its reasoning for `CONVERGING` with the counter-measures intact:

> both round-9 apparatus findings landed in constructs that no longer exist (the
> scanner is deleted) or are structurally repaired (`str.__eq__` bypasses caller
> dispatch), and rounds 8 and 9 each ended by **removing** a claim no
> implementation could keep. The finding-generating surface itself shrank — the
> inverse of a periphery growing one level deeper.

## The open-population question — it found two, and both were real

Both were reproduced here before anything was changed. Both are the class round 9
deleted, and both sit in rounds-1/2 apparatus that the previous two monitors had
placed out of scope, so naming them was a finding to reconcile rather than a
scope decision.

**1. `SMART_PARAM_TARGETS` was closed against a *spelling* of an assignment.**
The scan matched `resolved["<literal>"] = ...` inside two named functions. The
claim it was said to keep is stated in **production**, at `smart_params.py`: *a
new smart parameter cannot quietly start overwriting a fourth native name.*
Measured, not read — a fourth name added through `resolved.update({...})`:

```
scan sees the new name: False
test verdict:           PASSED with the assignment hidden
```

**Fixed by executing instead of parsing.** Both resolvers are now **run**, with
every smart parameter the provider declares switched on, across every task, and
the names they actually return are compared with the table. Running the code has
no spelling to guess. The input population is closed because it is enumerable,
and a separate test asserts the activation table equals
`LGBMProvider().smart_param_names()`, so a new smart parameter fails for having
no activation rather than passing unobserved. The production docstring now
describes what the test does.

**2. `_declared_writers` matched `ast.FunctionDef` in one class body.** An
`async def`, an assignment, a decorated or an inherited writer is absent from
that side *and* from the map, so the equality holds and the writer is never
substituted. Measured: making `export_code` an `async def` left the scan
returning `['export']`.

**Fixed by asking the class instead of its source.** `dir(Model)` sees async,
assigned, decorated and inherited members alike. The one assumption left is
stated where it lives: a writer whose name does not begin with `export` is not
covered, and no check can find it, because "writes to disk" is not a property of
a name.

RED verified for both: the execution-based check catches a name written through
`update` that the source scan missed, and the runtime lookup sees an async writer
that the source scan missed.

## Its recommendation, adopted

> **`redirect`** — the maintainer's premise still holds on this record (19/19
> findings reproduced), so continue, but under monitor 78's standing constraint
> (repair in place or delete; no new module, generator or scanner) *and* with the
> two scans above admitted for bounding or deletion, because leaving known
> instances of the class the last two rounds each removed is how the next finding
> gets manufactured.

Adopted in full, including the standing constraint: both repairs replace an
existing check in place — one parser became an execution, one parser became an
attribute lookup — and no module, generator or scanner was added.

It also recorded what is *not* this class: `values_differ`'s "does not raise an
`Exception`" is closed over the five expressions in the function body, not over
values.

Full suite **2403 passed**.
