# H-0130 — options for the "Facade-only assembly" rule

Status: options paper for the maintainer's decision (2026-10-15). It is reviewed by Codex for factual errors before the decision.

## 1. The problem

BLUEPRINT §2.1 (dependency rules, third bullet, `BLUEPRINT.md:58`) says that concrete-class assembly and type dispatch belong to Layer 4 (the Facade). Since H-0130 amendment 1 it names two exceptions:

1. A §2.2 provider's own factories and its checks on its own adapter type.
2. The two Layer 1 registries, `metrics/registry.py` and `calibration/registry.py`, building registered types by name.

The maintainer asked for this exception list to be enforced by a test, including type dispatch (option c). I added R6 (`tests/test_architecture/_dispatch_rules.py`, commit `fe001bd`). It does two things:
- it counts a closed set of dispatch spellings against an exact-count list;
- it limits registry attributes to the two registry modules.

Two Codex rounds then showed that R6 cannot keep its promise.

- **R6 is an open enumeration.** Run `h0130-amend1-r6-verify` found that it misses:
  - constructor calls;
  - chained comparisons (`_targets` accepts only one-operator `Compare` nodes);
  - registry access through an assignment alias, a re-export or `getattr`.

  Each fix adds one more spelling. Earlier rounds of this proposal had the same shape with other open lists.
- **The rule itself is undefined.** Run `h0130-assembly-critique` reproduced an AST census of constructor calls in Layers 1-3 to resolvable lizyml classes:

  | Where the class is defined | Calls |
  |---|---|
  | Layer 0 | 141 |
  | A declared interface module | 6 |
  | A concrete class of another category | 1: `estimators/lgbm/provider.py:365` builds `NativeFeaturePipeline`, which exception 1 covers |
  | A concrete class of the same category | 78: for example the eight `MethodSpec` calls from `calibration/_optimizer.py:57`, `CategoricalEncoder` / `FeatureTransformer` at `features/pipelines_native.py:40-41`, and the provider's adapter at `estimators/lgbm/provider.py:349` |

  Read literally, "only the Facade assembles concrete classes" is therefore false in 78 places. The intended meaning is presumably narrower: choosing and wiring the concrete implementation of a Layer 1 interface. BLUEPRINT does not define "assembly" that way. The sentence predates H-0130; H-0130 only added the exceptions.

## 2. Measured facts the options rely on

- **The import graph is already ownership-shaped.** `probes/impl_owner_census.py` → `.out` finds 13 modules defining a class that implements a Layer 1 interface (`BaseSplitter`, `BaseFeaturePipeline`, `BaseEstimatorAdapter`, `BaseMetric`, `BaseCalibratorAdapter`). Their importers fall into four groups only:
  - the category's own package `__init__.py` (public re-export: `lizyml.splitters`, `lizyml.metrics`, `lizyml.calibration`, `lizyml.estimators.lgbm`);
  - the category's registry module (`metrics/registry.py`, `calibration/registry.py`);
  - the LightGBM provider (`estimators/lgbm/provider.py` imports `estimators/lgbm/adapter.py` and `features/pipelines_native.py`);
  - the Facade (`core/_model_factories.py` imports the six splitter modules and `calibration/isotonic.py`).

  No Layer 2 or 3 module imports an implementation module. R3 already forbids that.
- **Registry names are used in four ways only** (verified by the `h0130-assembly-critique` reviewer):
  - `_Registry` is created only in `core/registries.py:12`, `:48` and `:49`;
  - the implementations call only `.register`: `calibration/{beta,isotonic,platt}.py` and `metrics/{classification,regression}.py`;
  - the two registry modules call `get`, and `keys` in the metrics registry;
  - the Facade calls `CalibratorRegistry.get` (`core/_model_factories.py:1262`, `:1265`).
- **Concrete classes are public.** `lizyml/metrics/__init__.py` and `lizyml/calibration/__init__.py` export concrete metric and calibrator classes. `lizyml.calibration.get_calibrator` is public.

## 3. Options for the rule and its enforcement

Each option says what it closes, what stays outside as a declared bound, and what it changes. "In H-0130" means it fits the accepted proposal plus amendment 1. Anything else needs a new decision by the maintainer and a proposal revision or a new proposal.

### O1. Keep extending R6 (syntax enumeration)

Add constructor calls, chained comparisons, assignment aliases, re-exports and `getattr` handling, then verify again.

- PRO: Keeps the current direction. No spec change beyond amendment 1. Size M.
- CON:
  - The target grammar is open. Constructors, comparison shapes, assignment flow and reflection have no natural end, which is the pattern this proposal's earlier rounds showed.
  - Counting constructors would put the 78 same-category calls on an exact-count list, which documents noise rather than a rule.
  - Each review round is likely to find one more form.
- Fits H-0130: yes, but it is unlikely to converge.

### O2. Close by name for registries only; declare dispatch and construction review-only

- How it works: as R5 does for `import_module`. The identifiers `MetricRegistry`, `CalibratorRegistry` and `_Registry` (as a name, an attribute, an import alias or a string constant) may appear only in listed modules. In the implementation modules they may appear only as `X.register`. Type dispatch and in-category construction are not tested, and §2.1 says they are reviewed.
- PRO:
  - Closed for registry access by construction: an alias, a re-export or a `getattr` with a constant string all need the identifier to appear.
  - Small (S).
  - No code change. Fits amendment 1 with a narrower R6.
- CON:
  - Computed strings (`getattr(m, "Metric" + "Registry")`), `vars()` and `__dict__` remain declared bounds, as for R5.
  - Hand-rolled registries (a dict of classes, an if-chain) are not caught.
  - It does not enforce type dispatch or assembly, so it does not meet the maintainer's stated goal for option c.
- Fits H-0130: yes (amendment 1, R6 narrowed).

### O3. Remove R6; the exception list is guarded by review

- PRO: Removes the false guarantee. Smallest (S). Same enforcement as before H-0130 amendment 1.
- CON: No automated regression boundary at all for the exception list.
- Fits H-0130: yes.

### O4. Define "assembly" and enforce it by import ownership (in place)

- How it works:
  - BLUEPRINT defines assembly as selecting and wiring the concrete implementation of a Layer 1 interface.
  - A table declares the implementation modules (today the 13 above) and, for each, its owners: the category's package `__init__` (public re-export), its registry module, the provider (for its adapter and pipeline), and the Facade.
  - R2/R3 gain an "owner" check: any other importer of an implementation module fails.
- PRO:
  - Using a concrete implementation in any spelling (constructor, comparison, `match`, alias) needs an import of its module. The existing import scan (R0-R5, every import statement) therefore closes it, with no expression grammar.
  - The 78 same-category helper constructions are not implementations of a Layer 1 interface, so the rule no longer contradicts them.
  - Today's graph already matches (section 2), so no code moves.
  - Public APIs unchanged.
- CON:
  - It changes the spec of an accepted proposal: a new definition and a new table in §2.1.
  - Bounds remain:
    - `eval` / `exec`;
    - what a package `__init__.py` loads at run time;
    - reflection on an instance already held, such as `type(obj)(...)`;
    - an implementation class defined outside the declared modules. That last one needs its own rule ("a class implementing a Layer 1 interface lives only in a declared module"), checkable by AST over class bases, which `impl_owner_census.py` does.
  - Third-party implementations outside `lizyml` are not covered and need not be.
  - Size M.
- Fits H-0130: only with a further revision (amendment 2) and its review, or as a new proposal.

### O4b. O4 with a structural layout (`_impl/` packages)

- PRO: Ownership is visible in the tree, and the owner table shrinks to package prefixes.
- CON:
  - It moves many modules, so pickled class paths of saved artifacts change for moved implementations (adapters and calibrators are pickled). That needs re-exports or a `format_version` decision.
  - It changes the documented provider layout (§2.2, `docs/add-estimator-guide.md`).
  - Size L.
- Fits H-0130: no; it needs a new proposal.

### O5. Capability-separated registries

- How it works:
  - Replace the globally importable `_Registry` objects with private per-category storage.
  - Implementation modules receive only a registration decorator. Only the two registry modules can look classes up.
  - The Facade's `CalibratorRegistry.get` becomes a calibration helper.
- PRO:
  - Lookup is unreachable from elsewhere: aliases, re-exports, `getattr`, `vars` and `__dict__` have no global object to reach.
  - The public `get_metric` / `get_metrics_for_task` / `get_calibrator` are unchanged.
  - Size M.
- CON:
  - Code change in `core/registries.py`, two registry modules, the implementation decorators and one Facade call.
  - Anyone importing `lizyml.core.registries` directly would break. It is not documented as public, so this needs checking.
  - Monkey-patching private modules stays a bound.
- Fits H-0130: no; it needs amendment 2 or a new proposal.

### O6. Registries as the only way to reach any Layer 1 implementation

Register typed factories for splitters, pipelines, providers or adapters, calibrators and metrics.

- PRO: Centralises all selection.
- CON:
  - To be literal, the public concrete exports (`lizyml.metrics`, `lizyml.calibration`) would have to be deprecated.
  - The estimator extension contract changes.
  - Size L.
- Fits H-0130: no.

### O7. Runtime construction authority

Implementations of Layer 1 interfaces can be constructed only inside a Facade, provider or registry context.

- PRO: Catches executed construction in any spelling.
- CON:
  - It cannot prove paths that are not executed.
  - It can be bypassed by monkey-patching.
  - Enforced at run time, it would break users who construct public `RMSE` or `PlattCalibrator` directly. Enforced in tests only, it depends on coverage.
  - Size M/L.
- Fits H-0130: no.

### O8. Typed factory boundary plus a type-checker plugin

- PRO: Improves the typed provider boundary. No runtime change.
- CON:
  - `Any`, casts and reflection escape it.
  - A mypy plugin is new tooling.
  - It reinforces another option; it cannot be the primary guard.
  - Size M.
- Fits H-0130: no.

## 4. Process options (where the work happens)

| Path | What happens | PRO | CON |
|---|---|---|---|
| P1 | Finish H-0130 with O2 or O3 now. O4 (+ O5) becomes a new proposal (H-0131). | #362's goal (the seventeen exceptions) is already met. The PR can merge soon. The new definition gets its own review with a closed scope. | §2.1's assembly sentence stays review-only until H-0131 lands. The literal 78-place contradiction stays until then, unless P1 also adds a one-line "assembly means wiring Layer 1 interface implementations" note, which is itself a spec change. |
| P2 | Add amendment 2 to H-0130 for O4 (+ O5), review it, and implement it on this branch. | One PR delivers the strongest design. | H-0130 has already needed three proposal rounds, an acceptance review and two amendment critiques. A further revision makes the branch and the review larger. #362 closes later. |
| P3 | O2 or O3 only; no follow-up. | Least work. | The assembly rule stays undefined and review-only permanently. |

## 5. My recommendation (for the maintainer to accept or reject)

P1:
1. Finish H-0130 with O2: R6 narrowed to name-based registry access.
2. Rewrite BLUEPRINT §2.1 and amendment 1 to say that type dispatch and construction are reviewed, not tested.
3. Fix the two minor findings: the census probe's import path and the amendment's "tests unchanged" sentence.
4. Open H-0131 for O4 + O5, starting from the facts in section 2.

O3 instead of O2 is the alternative if a partial registry guard is judged not worth its upkeep.

## 6. Evidence

- `probes/dispatch_census.py` / `.out`: dispatch spellings.
- `probes/impl_owner_census.py` / `.out`: implementation modules and their importers.
- `probes/construct_census.py` / `.out`: the 141 / 6 / 1 / 78 constructor census and the registry-name uses. The `h0130-assembly-critique` reviewer reproduced it independently.
- Codex runs: `h0130-amend1-critique`, `h0130-amend1-r6-verify`, `h0130-assembly-critique` (run directories under `~/.local/state/lizy-runner/runs/`).
