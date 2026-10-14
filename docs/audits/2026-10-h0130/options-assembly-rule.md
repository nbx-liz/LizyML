# H-0130 — options for the "Facade-only assembly" rule

Status: options paper for the maintainer's decision (2026-10-15).
- Revision 1 answers the Codex review `h0130-options-review` at `db6ca33`, which found 5 critical, 3 major and 2 minor errors. Each one is corrected below.
- A confirmation review follows before the decision.

## 1. The problem

BLUEPRINT §2.1 (dependency rules, third bullet, `BLUEPRINT.md:58`) says that concrete-class assembly and type dispatch belong to Layer 4 (the Facade). Since H-0130 amendment 1 it names exactly two exceptions:

1. A §2.2 provider's own factories, and its checks on its own adapter type.
2. The two Layer 1 registries, `metrics/registry.py` and `calibration/registry.py`, building registered types by name.

The maintainer asked for this exception list to be enforced by a test, including type dispatch ("option c"). Amendment 1 records that requirement as **R6**:
- `HISTORY.md`, H-0130 amendment 1;
- `BLUEPRINT.md` §2.1, "表の読み方と検査";
- `tests/test_architecture/_dispatch_rules.py`, with `DISPATCH_ALLOWED` in `test_layer_rules.py`.

R6 counts a closed set of dispatch spellings against an exact-count list, and limits registry attributes to the two registry modules. **Any option below that does not enforce dispatch therefore withdraws an accepted requirement of amendment 1. It needs a maintainer decision and an amendment 2 (or a new proposal) with its own review. Amendment 1 is not to be rewritten.**

Two Codex rounds showed that R6 cannot keep its promise.

- **R6 is an open enumeration.** Run `h0130-amend1-r6-verify` found that it misses:
  - constructor calls;
  - chained comparisons (`_targets` accepts only one-operator `Compare` nodes);
  - registry access through an assignment alias, a re-export or `getattr`.

  Each fix adds one more spelling. Earlier rounds of this proposal had the same shape with other open lists.
- **The rule is undefined.** An AST census of constructor calls in Layers 1-3 to resolvable lizyml classes (`probes/construct_census.py` → `.out`, reproduced independently by run `h0130-assembly-critique`) gives:

  | Where the class is defined | Calls |
  |---|---|
  | Layer 0 | 141 |
  | A declared interface module | 6 |
  | A concrete class of another category | 1: `estimators/lgbm/provider.py:365` builds `NativeFeaturePipeline`, which exception 1 covers |
  | A concrete class of the same category | 78 |

  - Most of the 78 are ordinary helpers: for example the eight `MethodSpec` calls from `calibration/_optimizer.py:57`, and `CategoricalEncoder` / `FeatureTransformer` at `features/pipelines_native.py:40-41`.
  - At least one is an interface implementation: the provider building its own `LGBMAdapter` at `estimators/lgbm/provider.py:349`, which exception 1 covers.
  - Read literally, "only the Facade assembles concrete classes, with two exceptions" is false for the helper calls. The intended meaning is presumably narrower: choosing and wiring the concrete implementation of a Layer 1 interface. BLUEPRINT does not define "assembly" that way. The sentence predates H-0130; H-0130 only added the exceptions.

## 2. Measured facts the options rely on

### 2.1 Implementations of Layer 1 interfaces

`probes/impl_owner_census.py` → `.out` covers **inherited** component interfaces only. It finds 13 modules defining a class whose bases resolve, transitively, to `BaseSplitter`, `BaseFeaturePipeline`, `BaseEstimatorAdapter`, `BaseMetric` or `BaseCalibratorAdapter`.

Their direct importers fall into four groups:
- the category's package `__init__.py`, which re-exports them publicly (`lizyml.splitters`, `lizyml.metrics`, `lizyml.calibration`, `lizyml.estimators.lgbm`);
- the category's registry module (`metrics/registry.py`, `calibration/registry.py`);
- the LightGBM provider (`estimators/lgbm/provider.py` imports `estimators/lgbm/adapter.py` and `features/pipelines_native.py`);
- the Facade (`core/_model_factories.py` imports the six splitter modules and `calibration/isotonic.py`).

No Layer 2 or 3 module imports an implementation module directly; R3 already forbids it.

The census does **not** cover structural implementations:
- `EstimatorProvider` is a Protocol (`estimators/provider.py:168`), implemented without inheritance by `LGBMProvider` (`estimators/lgbm/provider.py:256`).
- The Facade imports and constructs it at `core/_model_factories.py:495-497`.
- Any owner table built from this census must add providers separately. A rule that finds implementations by AST bases would miss future structural providers.

### 2.2 Public re-exports of implementations

Concrete implementations are re-exported by package `__init__` files and are public:
- `lizyml.splitters`: eight concrete splitters.
- `lizyml.metrics`: the concrete metrics, imported from `classification` and `regression`.
- `lizyml.calibration`: the concrete calibrators.
- `lizyml.estimators` and `lizyml.estimators.lgbm`: `LGBMAdapter`.

`lizyml.calibration.get_calibrator` is public too.

The H-0129 checker (`_layer_rules.resolve`) records `from lizyml.metrics import RMSE` as an edge to `lizyml.metrics`, not to `lizyml.metrics.regression`. So an import of a re-exported implementation through its package does not show the defining module.

### 2.3 Registries

These locations were verified by `h0130-assembly-critique` and `h0130-options-review`:
- `_Registry` is defined at `core/registries.py:12` and instantiated twice, at `:48` and `:49`.
- The implementations call only `.register`: `calibration/{beta,isotonic,platt}.py` and `metrics/{classification,regression}.py`.
- The two registry modules call `get`; the metrics registry also calls `keys`.
- The Facade imports `CalibratorRegistry` at `core/_model_factories.py:1262` and calls `.get` once, at `:1265`.

`lizyml.core.registries` is not re-exported by `lizyml` or `lizyml.core`, and no user guide imports it. BLUEPRINT §2.1, however, lists `core/registries.py` as a Layer 0 module every layer may import, and the BLUEPRINT and ARCHITECTURE trees name its symbols. It is therefore a documented architectural interface, though not an advertised end-user API.

### 2.4 What is pickled

- Estimator adapters: `FitResult.models` and `RefitResult.model`.
- The concrete calibrator inside `CalibrationResult.c_final` (in `FitResult.calibrator`).
- Not persisted by these result objects: pipeline instances (only their state dicts are saved), splitters, metrics and providers.

`persistence/loader.py` checks `format_version` and then calls `joblib.load`. Migration (`_migrate_fit_result`, around `loader.py:202`) runs only after unpickling.

## 3. Options for the rule and its enforcement

"Fits H-0130" means it fits the accepted proposal plus amendment 1 as written. Everything else needs a maintainer decision and an amendment 2 or a new proposal, with review.

### O1. Keep extending R6 (syntax enumeration)

Add constructor calls, chained comparisons, assignment aliases, re-exports and `getattr` handling, then verify again.

- PRO: Keeps amendment 1's mechanism. Size M for the grammar alone.
- CON:
  - The target grammar is open. Constructors, comparison shapes, assignment flow and reflection have no natural end, which is the pattern this proposal's rounds showed.
  - Checking constructors against the literal rule makes most of the 78 helper calls violations. They can either fail the tree, or pass through an allowlist or a narrower definition of assembly. Either way that is a specification change, not just a test change.
- Fits H-0130: only for the dispatch part. Adding constructors needs a decision on the definition, which means an amendment.

### O2. Close by name for registries only; declare dispatch and construction review-only

- How it works:
  - As R5 does for `import_module`: the identifiers `MetricRegistry`, `CalibratorRegistry` and `_Registry` (as a name, an attribute, an import alias or a string constant) may appear only in listed modules.
  - In the implementation modules they may appear only as `X.register`.
  - Type dispatch and in-category construction are not tested, and §2.1 says they are reviewed.
- PRO:
  - Registry access is closed by construction: an alias, a re-export or a `getattr` with a constant string all need the identifier.
  - Size S for the code.
- CON:
  - It **withdraws amendment 1's accepted dispatch requirement**. It needs a maintainer decision, an amendment 2 with review, and changes to BLUEPRINT §2.1, the acceptance criteria, the R6 tests and mutations, and the negative controls.
  - Computed strings, `vars()` and `__dict__` remain declared bounds, as for R5.
  - Hand-rolled registries (a dict of classes, an if-chain) are not caught.
  - It does not meet the maintainer's stated goal for option c.
- Fits H-0130: no; amendment 2.

### O3. Remove R6; the exception list is guarded by review

- PRO: Removes the false guarantee. Size S for the code.
- CON:
  - Like O2, it withdraws amendment 1's requirement and needs amendment 2.
  - No automated regression boundary for the exception list.
  - The literal Facade-only sentence stays in force and stays undefined.
- Fits H-0130: no; amendment 2.

### O4. Define "assembly" and enforce it by import ownership (in place)

- How it works:
  - BLUEPRINT defines assembly as selecting and wiring the concrete implementation of a Layer 1 interface.
  - A table declares the implementation modules and, for each, its owners:
    - the 13 inherited ones in section 2.1;
    - the provider module(s), declared explicitly because providers implement a Protocol structurally;
    - owners are the category's package `__init__` (public re-export), its registry module, the provider (for its adapter and pipeline), and the Facade.
  - An owner check is added to the import rules.
- PRO:
  - Ordinary same-category helper calls (most of the 78) are not implementations of a Layer 1 interface, so they are outside the rule. The one implementation constructed in the 78, the provider's `LGBMAdapter`, is an owner use.
  - Direct imports already match the owner table (section 2.1), and public APIs stay unchanged.
- CON:
  - **Not closed as described until re-exports are handled.** The existing resolver records an import through a package (`from lizyml.metrics import RMSE`, `from lizyml.splitters import KFoldSplitter`, `from lizyml.estimators import LGBMAdapter`) as an edge to the package, not to the defining module (section 2.2). `estimators` -> `metrics` is an allowed pair, so a non-owner in `estimators/` could use `RMSE` through `lizyml.metrics` unnoticed. The design must either:
    - resolve imported names through re-exports to their defining module; or
    - forbid non-owners in Layers 1-3 from importing implementation names from the re-exporting packages.

    Both are new checker logic with their own grammar to close.
  - The owner table must list structural implementations (providers) explicitly. "Find implementations by AST bases" is not complete (section 2.1).
  - Bounds remain:
    - `eval` / `exec`;
    - what a package `__init__.py` loads at run time;
    - reflection on an instance already held, such as `type(obj)(...)`.
  - It changes the spec of an accepted proposal: a new definition and table in §2.1.
  - Size M, possibly M/L with the re-export resolution.
- Fits H-0130: no; amendment 2 or a new proposal.

### O4b. O4 with a structural layout (`_impl/` packages)

- PRO: Ownership is visible in the tree, and the owner table shrinks to package prefixes.
- CON:
  - Moving the implementation modules changes the class path recorded in pickles for the persisted objects: adapters (`FitResult.models`, `RefitResult.model`) and calibrators (`CalibrationResult.c_final`). Section 2.4.
  - A `format_version` bump cannot help, because `joblib.load` runs before migration. A package-level re-export is not enough either. Compatibility needs one of three things:
    1. shim modules at every old defining path;
    2. a custom unpickler that remaps old paths;
    3. an explicit decision to reject artifacts saved before the move.
  - It changes the documented provider layout (§2.2, `docs/add-estimator-guide.md`).
  - Size L.
- Fits H-0130: no; a new proposal.

### O5. Capability-separated registries

- How it works:
  - Replace the globally importable `_Registry` objects with private per-category storage.
  - Implementation modules receive only a registration decorator. Only the two registry modules can look classes up.
  - The Facade's `CalibratorRegistry.get` becomes a calibration helper.
- PRO:
  - Lookup is unreachable from elsewhere: aliases, re-exports, `getattr`, `vars` and `__dict__` have no global object to reach.
  - Public `get_metric`, `get_metrics_for_task` and `get_calibrator` are unchanged.
  - Size M.
- CON:
  - It removes or changes `MetricRegistry` / `CalibratorRegistry` in `core/registries.py`, which BLUEPRINT documents as a Layer 0 interface (section 2.3). That needs a compatibility decision (a deprecation shim or a documented break) for direct importers.
  - Code changes in the registry modules, the implementation decorators and one Facade call.
  - Monkey-patching private modules stays a bound.
- Fits H-0130: no; amendment 2 or a new proposal.

### O6. Registries as the only way to reach any Layer 1 implementation

Register typed factories for splitters, pipelines, providers or adapters, calibrators and metrics.

- PRO: Centralises all selection.
- CON:
  - To be literal, every public concrete export would have to be deprecated or redirected: the eight splitters in `lizyml.splitters`, the metrics in `lizyml.metrics`, the calibrators in `lizyml.calibration`, `LGBMAdapter` in `lizyml.estimators` and `lizyml.estimators.lgbm`, and their direct module paths. That is a broad public API change across five packages.
  - The estimator extension contract changes.
  - Size L.
- Fits H-0130: no; a new proposal.

### O7. Runtime construction authority

Implementations of Layer 1 interfaces can be constructed only inside a Facade, provider or registry context.

- PRO: Catches executed construction in any spelling.
- CON:
  - It cannot prove paths that are not executed.
  - It can be bypassed by monkey-patching.
  - Enforced at run time, it would break users who construct public splitters, metrics or calibrators directly. Enforced in tests only, it depends on coverage.
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

### O9. Remove the universal assembly sentence; keep the enforced import rules

- How it works:
  - Delete the Facade-only sentence from BLUEPRINT §2.1 (and its copies in ARCHITECTURE.md and README).
  - Keep R0-R5, which already stop any Layer 2/3 code, and any undeclared Layer 1 category, from importing concrete implementations.
  - Describe the provider factories and the registries as how the code is built, without claiming exclusive construction authority.
  - Remove R6.
- PRO:
  - The specification states only what is enforced. The 78-place literal contradiction disappears.
  - No new checker. Size S/M for the documents.
- CON:
  - It gives up a design intent: "choosing implementations is the Facade's job" is no longer stated at all, so in-category wiring drift is neither forbidden nor reviewed against a rule.
  - It withdraws amendment 1's R6 requirement, and also changes H-0130's accepted exception text.
  - Documentation churn across BLUEPRINT, ARCHITECTURE, README and the guide, which H-0130 has just rewritten.
- Fits H-0130: no; amendment 2.

## 4. Process options (where the work happens)

| Path | What happens | PRO | CON |
|---|---|---|---|
| P1 | Finish H-0130 with amendment 2 choosing O2 (or O3 / O9). Then O4 (+ O5) becomes a new proposal (H-0131). | #362's goal (the seventeen exceptions) is already met and stays met. The larger definition gets its own proposal and review with a closed scope. | Amendment 2 is a maintainer decision plus its own review, and it rewrites BLUEPRINT §2.1, the acceptance criteria, the R6 tests and mutations, and the negative controls. Until H-0131 lands, dispatch and assembly are review-only, and the literal sentence stays undefined unless O9 removes it. |
| P2 | Amendment 2 to H-0130 for O4 (+ O5), reviewed, then implemented on this branch. | One PR delivers the strongest design. | O4 still needs re-export resolution and an explicit provider list (section 3). H-0130 has already needed three proposal rounds, an acceptance review, two amendment critiques and this paper. The branch and the review grow, and #362 closes later. |
| P3 | Amendment 2 choosing O2, O3 or O9, with no follow-up proposal. | Least work. | The assembly rule stays review-only, or is gone (O9), permanently. |

## 5. My recommendation (for the maintainer to accept or reject)

P1 with O2:
1. Amendment 2 withdraws amendment 1's dispatch requirement and narrows R6 to name-based registry access.
2. Its review covers the rewritten §2.1, the acceptance criteria, the R6 tests and mutations, and the negative controls.
3. It fixes the two minor findings of `h0130-amend1-r6-verify`: the census probe's import path and the "tests unchanged" sentence.
4. H-0131 follows for O4 + O5, starting from section 2.

This costs one more decision and one more review on H-0130, which section 4 does not hide. O9 is the alternative if the maintainer prefers the specification to claim only what is enforced, at the price of dropping the stated design intent.

## 6. Evidence

- `probes/dispatch_census.py` / `.out`: dispatch spellings.
- `probes/impl_owner_census.py` / `.out`: inherited implementations of Layer 1 interfaces and their importers. It does not list structural providers.
- `probes/construct_census.py` / `.out`: the 141 / 6 / 1 / 78 constructor census and the registry-name uses.
- Codex runs (run directories under `~/.local/state/lizy-runner/runs/`):
  - `h0130-amend1-critique`
  - `h0130-amend1-r6-verify`
  - `h0130-assembly-critique`
  - `h0130-options-review`
