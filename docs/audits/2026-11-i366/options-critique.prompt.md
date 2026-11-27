Review-kind: critique

# Critique: which BLUEPRINT §2.1 assembly and dispatch concerns can tests enforce, and how? (#366, before Proposal H-0131 is drafted)

This is a one-off design analysis, not a review of a diff. No Proposal text exists yet. The owner asked for an independent analysis of the candidates before choosing a direction (#366 issuecomment-7208403717). Answer as findings, each with a `file:line` location and evidence. Use `CHANGES_REQUESTED` when something needs action and `APPROVE` when nothing does. The findings are the analysis; the verdict is not.

## The situation

- **The amendment.** H-0130 amendment 2 (`HISTORY.md`, `## H-0130`, section `### 改訂 2`, near line 14126) defines two rules in BLUEPRINT §2.1 (`BLUEPRINT.md` from line 43):
  - *concrete assembly*: selecting the concrete implementation of a Layer 1 boundary interface from a declarative input and wiring it to its user;
  - *concrete type dispatch*: branching on the identity of a concrete implementation or of a concrete cross-category result type.

  Both belong to the Facade (Layer 4), with two exceptions: a provider's own factories and checks on its own adapter, and the metric and calibrator registry helpers.
- **What the tests check today.** Amendment 1's syntax-enumeration check (R6) proved open and was withdrawn. The layer tests (`tests/test_architecture/_layer_rules.py`, `test_layer_rules.py`, `test_h0130_contracts.py`) check static import edges only (R0–R5).
- **What review checks today.** Four concerns are review-only, through the PR template:
  1. concrete assembly, including selection by Config type, discriminator `name`, or a derived value;
  2. concrete type dispatch;
  3. registry lookups that bypass the registry helpers;
  4. rewriting registrations (an implementation module registering its own types is allowed).
- **#366 asks for a Proposal (H-0131)** that decides which of the four the tests should enforce and how. Its candidates are:
  - **O4**: import ownership; only declared owners may import an implementation module.
  - **O5**: capability-separated registries; implementations get only a registration capability, and only the registry helpers can look up.
  - **1-C**: an interface module exports only the names it declares, e.g. `__all__`.

  It lists ten scope items the Proposal must decide: the role table, the component universe, freshness, re-exports, dynamic import and reflection bounds, internal vs users, third-party providers, registries and compatibility, selection by name or value, and the end condition. The issue text is pasted verbatim at the end of this prompt. Also read the options paper `docs/audits/2026-10-h0130/options-assembly-rule.md` (sections 2–5), which carries the measured facts (implementation census, re-export behaviour of `_layer_rules.resolve`, registry call sites, what is pickled). The census probes and their outputs are in `docs/audits/2026-10-h0130/probes/`.
- **History of the loop.** H-0130 needed three proposal rounds, an acceptance review, two amendment critiques and an options paper. The repeated failure was a check over an open grammar, where each round added one more spelling. The owner wants H-0131's review loop to have a fixed, closable subject.

## Working hypothesis (test it; do not assume it)

- O5 closes concern 3, and with a registration-only capability also concern 4, by construction: there is no global object to reach.
- O4 plus 1-C closes "a non-owner in Layers 1–3 imports an implementation", provided re-exports are handled by forbidding non-owners from importing implementation names from re-exporting packages, rather than by resolving names through re-exports.
- Selection by name or value (scope item 9), and concrete type dispatch (concern 2), have an open grammar and should stay with review.

## Questions

1. **Closability.** For each of concerns 1–4, which candidate (O4, O5, 1-C, a combination, or none) closes it by construction or by a closed grammar, and which would reopen the R6 pattern? For each "closed" claim, name the bound: the grammar, schema or explicit enumeration that closes it. Name what stays outside the bound (e.g. `eval`, `getattr` with computed strings, `type(obj)(...)`, monkey-patching). Point to the code that makes the claim true or false today.
2. **Re-exports.** Which design closes re-exports (multi-hop, aliases, `*`, module-attribute access such as `import lizyml.metrics as m; m.RMSE`)? Choose between resolving through re-exports and forbidding implementation names from re-exporting packages for non-owners. What does 1-C add, and what happens when `__all__` is absent or stale? Point to `_layer_rules.resolve` and the package `__init__` files.
3. **Component universe and freshness.** How can a test find every implementation, including structural `Protocol` implementations such as `LGBMProvider` (`estimators/provider.py`, `estimators/lgbm/provider.py`) and concrete cross-category result types, so that a new unlisted implementation fails? Is there a closed way, or must providers be declared?
4. **O5 compatibility.** What exactly breaks for importers of `MetricRegistry` / `CalibratorRegistry` (`core/registries.py`, documented in BLUEPRINT as Layer 0)? Is a deprecation shim possible without leaving a global lookup reachable (which would undo O5)? What else touches pickles or the public API?
5. **Scope and cost.** Rank the plausible directions (for example O5 only; O5 + O4; O5 + O4 + 1-C; O4 + 1-C without O5; "keep review-only and close #366 with a documented bound"). For each give its size, its compatibility impact, which of concerns 1–4 end up test-enforced versus review-only (scope item 10), and the risk of an open review loop. Recommend one, precisely enough that the Proposal's acceptance criteria and review stop criterion can be written from it.
6. **Anything the ten scope items miss**, or that the options paper's measured facts no longer match at this head (the paper was written at H-0130 time, 2026-10-15; the tree has changed since: H-0138–H-0150).

## Environment

Read-only copy of this head (`develop` at the head given to the runner) without Git history. pandas, LightGBM and Optuna may be absent. AST-level probes with the standard library are fine. State what could not run.

## Issue #366 (verbatim)

````markdown
## Problem

H-0130 amendment 2 (#362) defines two rules in BLUEPRINT §2.1:

- **Concrete assembly**: selecting a concrete implementation of a Layer 1 boundary interface from a declarative input and wiring it to its user. This includes selection by Config type, the discriminator `name`, or a value derived from them.
- **Concrete type dispatch**: branching on the identity of a concrete implementation type or of a concrete cross-category result type.

Both belong to the Facade (Layer 4), with two exceptions:
- an estimator provider's own factories, and checks on its own adapter type;
- the metric and calibrator registry helpers, which build registered types by name.

Amendment 1's test of these rules (R6) was syntax-based and turned out to be open: constructors, chained comparisons, aliases, re-exports and `getattr` all escaped it. Amendment 2 therefore withdrew R6, and no replacement check was added. The layer test now checks **static import edges only** (R0-R5).

Until this proposal's implementation and the matching architecture tests are merged, the following are **checked by review**, through the PR template item:

1. concrete assembly, including selection by Config type, `name`, or a value derived from them;
2. concrete type dispatch;
3. registry lookups that bypass the registry helpers;
4. rewriting registrations. An implementation module registering its own types is allowed.

## Goal

Decide, in a Proposal (H-0131), which of these four the tests should enforce and how. The scope is closed below, so the review loop has a fixed subject.

## Candidates (none decided)

All three are from `docs/audits/2026-10-h0130/options-assembly-rule.md` and H-0130 amendment 2.

- **O4**: enforce by import ownership. Only declared owners may import an implementation module.
- **O5**: capability-separated registries. Implementations receive only a registration capability, and only the registry helpers can look up.
- **1-C**: an interface module may export only the names it declares, for example in `__all__`.

## Scope: decisions H-0131 must make

These are the ten items of H-0130 amendment 2.

1. **A role table, not a flat owner table.**
   - Package `__init__` modules re-export only.
   - Implementation modules register only.
   - Registry helpers look up and construct.
   - Each provider builds and checks only its own adapter and pipeline.
   - Named composition modules in the Facade select and wire.
2. **The component universe.** It includes structural `Protocol` implementations (`EstimatorProvider`) and concrete cross-category result types.
3. **Freshness.** A new implementation that is not in the table must fail.
4. **Re-exports.** Multi-hop re-exports, aliases, `*` re-exports and module-attribute access. If 1-C is chosen, the Proposal must also decide:
   - where each interface's declaration lives;
   - what happens when `__all__` is absent;
   - how staleness is detected;
   - one should-fail case per resolved path.
5. **Dynamic import and reflection bounds.**
6. **Internal vs. users.** Users may construct public concrete classes. The rule applies inside `lizyml` only.
7. **Third-party and optional providers.** How a provider outside the checked tree is discovered and trusted.
8. **Registries.**
   - Whether to adopt O5.
   - If O5 is adopted: compatibility and deprecation of the existing `MetricRegistry` and `CalibratorRegistry` symbols. `core/registries.py` is a documented architectural interface.
9. **Selection by name or value.** O4 and 1-C do not close these, because they rely on imports:
   - `cfg.name` or discriminator values;
   - class-name strings;
   - choosing between injected factories;
   - knowing concrete names, as `codegen/generator.py:140` does.

   Decide separately whether they enter the rules and the tests.
10. **End condition.** Merging the implementation and the matching architecture tests ends amendment 2's review-only period. The Proposal must state which of the four concerns are then enforced by tests and which stay with review.

## References

- `HISTORY.md`: H-0130, amendments 1 and 2.
- BLUEPRINT §2.1 and §2.2.
- `docs/audits/2026-10-h0130/options-assembly-rule.md`.
- Census probes in `docs/audits/2026-10-h0130/probes/`.
````
