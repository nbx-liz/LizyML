# H-0122 PR 2 design: `tune(space=...)`, `read_study_lineage`, `restore_tuning`, round boundaries on reporting surfaces, and the #342 fix

This is a read-only design; I changed no files. It is based on `fix/h0122-resume-compatibility` @ `4fc947d`, which is the same code PR 2 will branch from.

The section numbers §x.y refer to the spec, `docs/audits/2026-10-tuning-resume/resume-spec-draft.md`. It is the same text as HISTORY.md from line 12053.

PR 2 criteria covered here:
- In full: 4, 6, 6b, 7, 16a, 17, 20, 29, 34, 35, 36.
- In part:
  - 8b: the `space=` part.
  - 12: the category change.
  - 14: the `space=` rows.
  - 15: rounds that change the space with `space=`.
  - 19: the parts that pass `space=`.
  - 22: the refusal rows PR 2 adds.
  - 30: bullet 1.
- Also: the fix for #342.

**Decisions this design takes (each one is explained again in §9):**
- **D-a.** `StoredRound.space_arg` becomes a tuple of encoded dims instead of a raw dict.
- **D-b.** When a `space=` dim uses an alias of a parameter that already has a route, it is renamed to the routed raw name.
- **D-c.** `resume=False` with `space=` is refused in phase A, before any storage I/O. This settles PR 1's R7.
- **D-d.** `restore_tuning` stays a method on `ModelTuningMixin`. It does not get a fourth mixin.
- **D-e.** The extras registry gains a `restore_tuning` row. This needs a maintainer decision.

---

## 1. Module layout

### 1.1 Constraints found
- **Overlay seam scan.** `tests/test_core/test_fit_params_override.py:1804`: `SEAM_MODULES = (model.py, _model_tuning.py)`. The scan collects the layer names passed to bare `overlay_params(...)` calls in those two files, and the set must equal `SEAM_TESTS`, which is `{fixed, best_model_params, override, model_p}`.
  - `model_p` occurs only in `_build_tune_objective` (`_model_tuning.py:686`), so that method **must stay** in `_model_tuning.py`.
  - `_check_round_space` (`:604-641`) overlays the `fixed` layer. `model.py:555` overlays `fixed` too, so moving `_check_round_space` leaves the derived set unchanged.
- **Refusal matrix.** `tests/test_core/test_refusal_matrix.py:139-167` derives its columns from every `check_*` / `normalise_*` name in `_model_factories`. **Add no new `check_*` functions there.** Extend the existing ones with keyword arguments only.
- **Mixin isolation.** `tests/test_core/test_mixin_state_isolation.py:198-223` documents `_model_tuning.py` as the *sole* writer mixin. So `restore_tuning` stays on `ModelTuningMixin` and delegates to a pure helper (D-d).
- **Layering.** `lizyml/tuning/` is Layer 2 and cannot import `lizyml.core._tuning_resume` (Layer 4). `precheck_study_args` (`_tuning_resume.py:66-97`) therefore moves into `lizyml/tuning/tuner.py`. `_tuning_resume` re-exports it, so the existing import sites keep working.
- **800-line cap.** Two files are close: `lineage_codec.py` has 760 lines and `_model_tuning.py` has 791.

### 1.2 Files

| File | Layer | Status | Approx. lines after | Responsibility |
|---|---|---|---|---|
| `lizyml/core/_model_tuning.py` | 4 | rewire | 791 → ~725 | `tune(space=)`, the `restore_tuning` method, `_commit_tuning`, `_build_tune_objective` (unchanged), `_run_tune_round` (name kept: patched at `test_fit_params_override.py:3090`, `test_space_merge.py:369`), `_close_round`, `_get_tuning_state` |
| `lizyml/core/_tuning_admission.py` | 4 | **new** | ~190 | Moved here: `_validate_tune_inputs` (`:483-513`) as `validate_tune_args`; `_check_config_space_names` (`:515-548`) as `check_space_names(provider, cfg, *, space=None, surface=...)`; `_check_round_space` (`:604-641`) as `check_round_space(cfg, plan, provider, base_model, base_smart)`. New: `check_space_arg_shape(space)`, `refuse_space_on_create(...)`. The comments move with the code. |
| `lizyml/core/_tuning_space.py` | 4 | **new** | ~260 | §5.2: `Declaration`, `resolve_space`, `resolve_declaration`, `refuse_unchangeable` (5.2-3), `carry_forward` (5.2-2), `outside_range` (5.3 bullet 2), `same_declared_space` (U2) |
| `lizyml/core/_tuning_plan.py` | 4 | extend | 500 → ~620 | `plan_round` dispatch, moved from `_model_tuning.py:550-602`. `plan_next_round(declaration=)`. `plan_open_round(space_declaration=)`. `RoundPlan.source` / `.space_arg`. `resolve_config_space` becomes a wrapper over `resolve_space`. |
| `lizyml/core/_tuning_resume.py` | 4 | small | 431 → ~470 | `objective_identity(...)` and `judge_existing(...)` are extracted from tune's phase C (`:287-321`) and shared with `restore_tuning`. Re-exports `precheck_study_args`. |
| `lizyml/core/_tuning_restore.py` | 4 | **new** | ~110 | `restore_from_study(...) -> Restored`, pure (§4.2) |
| `lizyml/tuning/study_lineage.py` | 2 | **new** | ~150 | `StudyState`, `RoundRecord`, `StudyLineage`, `lineage_view` (pure), `read_study_lineage` |
| `lizyml/tuning/lineage_records.py` | 2 | **new (split)** | ~350 | `lineage_codec.py:420-760` moves here (record types, record codecs, `objective_digest`), with the `space_arg` changes |
| `lizyml/tuning/lineage_codec.py` | 2 | shrink | 760 → ~430 | Scalars, dims and reports. New `dim_signature(dim) -> str` (JSON of `encode_dim`). |
| `lizyml/tuning/lineage.py` | 2 | imports | ~455 | Re-exports from `lineage_records`, as it does today from `lineage_codec` (`:33-49`) |
| `lizyml/tuning/tuner.py` | 2 | +30 | ~460 | Receives `precheck_study_args`, unchanged |
| `lizyml/tuning/search_space.py` | 2 | fix | ~605 | #342 (§6) |
| `lizyml/tuning/__init__.py` | 2 | exports | — | `RoundRecord`, `StudyLineage`, `read_study_lineage` |
| `lizyml/plots/tuning.py` | 3 | rewrite `:89-118` | ~125 | Round spans come from trial rounds (§5) |
| `lizyml/core/_model_factories.py` | 4 | kwargs | — | `model_space_names(cfg, *, space=None, surface="tuning.optuna.space")`. `check_duplicate_space_dimensions` and `check_training_managed_space` gain the same two kwargs and quote `surface` in messages and context instead of the hard-coded `"tuning.optuna.space"` (`:924, :939, :777`). |
| `lizyml/_extras.py` | — | +1 row | — | `ExtraRule("restore_tuning", "tuning")` (D-e) |

### 1.3 Import fallout of the codec split
- `lineage_digest.py:9` docstring: points at `lineage_codec.decode_study`; change it to `lineage_records.decode_study`.
- `tests/test_tuning/test_resume_open_round.py:31`: imports `decode_round_end` from `lizyml.tuning.lineage_codec`; change it to `lizyml.tuning.lineage`.
- `_tuning_resume.py:41` and the imports in `test_lineage_codec.py:20-31` use only codec-level names, so they stay.

---

## 2. The `space=` flow

### 2.1 Signature

```python
def tune(self, data=None, *, resume=False, n_trials=None, expand_boundary=None,
         boundary_threshold=0.05, progress_callback=None, storage=None,
         study_name=None, space: Mapping[str, Any] | None = None) -> TuningResult
```

- `space` is keyword-only and comes last, so no positional call shifts.
- Its form is the same as `tuning.optuna.space`.
- `None` means "not given". `{}` is a real declaration: in merge mode it declares the provider defaults only, and in replace mode it declares an empty space.

### 2.2 Where `space=` enters PR 1's phases A–F (`_model_tuning.py:247-479`)

**Phase A** (no I/O). The steps run in this order after `validate_tune_args`:
- **A-s1.** `check_space_arg_shape(space)`: the value must be a `dict` with `str` keys whose values are `dict`s, and `category` (if present) must be one of `model` / `smart` / `training`. Otherwise `CONFIG_INVALID` with surface `tune(space=)`.
  - The config space gets no such check: `config/schema.py:556` types it as `dict[str, Any]`, and `parse_space` calls `spec.get` and trusts `category`.
  - An unknown category would otherwise reach `encode_dim` and fail late, as `_codec_refusal`.
- **A-s2.** `space is not None and not resume` → `CONFIG_INVALID`, `context={"surface": "tune(space=)", "reason": "creates_study"}`.
  - Every `resume=False` call creates a study (rows 1 and 2) or is refused (row 3), so this refusal is always correct.
  - It runs before `locate_study`, so `get_all_study_names` is never called and no SQLite file is created (criterion 14, "storage に何も書かれない"; D-c).
- **A-s3.** `space_dims = parse_space(dict(space))`. This raises `CONFIG_INVALID` for a malformed dim. The raw names are kept.

**Phase C** (`:287-321`, now `judge_existing`). Right after `refuse_state`:
- **C-s1.** `refuse_space_on_create(space_given, target.kind, lineage)`. Refuse when `target.kind == "create_storage"` (U3) or `lineage.state in {"uninitialized", "no_round"}` (E / N). This covers criteria 14 (U3) and 19 (N, and E per the §7.4 row E).
- A held in-memory study can only be in state O, C or F: a failed first tune drops it (`:411-417` and the commit at `:461-472`). So no other creation path exists.
- **R7 for U3.** `locate_study` → `load_stored_study` → `stored_study_exists` → `get_all_study_names(url)` builds an `RDBStorage`, which can create the SQLite file and schema. No study and no attribute is written, so INV-R4 (which is about the study) holds. Accept this and state it in 実装の記録 2. The U3 test asserts `study_name not in optuna.get_all_study_names(url)`, not that the file is absent.

**Phase D** (`:323-357`). The order is fixed so that the most specific message wins:

1. `check_space_names(provider, cfg)` on the config space. This is unchanged: the config space still equals `origin_space`, which C13a proves.
2. If `space_dims` is set: `declaration = resolve_declaration(space_dims, space_mode=lineage.study.space_mode, provider=provider, task=cfg.task, routes=lineage.routes, base_smart=None)`. This step also runs the §4.1b route check (§2.5). H-0121 runs later, in step 5, because it needs `base_smart`.
3. `check_space_names(provider, cfg, space=space, surface="tune(space=)")`. This is the 5.2-4 retarget of the name, duplicate-spelling and training-ownership checks (§2.6).
4. `_merge_params(provider, tuning_fixed_params=fixed, include_tuning_result=False)` (`:330-333`, unchanged).
5. `plan = plan_round(..., space_dims=space_dims, provider=provider, base_smart=base_smart)`. Inside it, `resolve_declaration` finishes with `admit_smart_dimensions`, and the planner applies 5.2-3, 5.2-2 and 5.3.
6. `check_round_space(cfg, plan, provider, base_model, base_smart)`. This is the old `_check_round_space`, run on `plan.effective`, so `validate_tuning_dimensions` and the training-claims checks now see the `space=` dims.
7. `round_record_payload(plan)`. The codec encodes `space_arg` (§2.7).

The INV-R4 boundary does not move: every new refusal is in phase A, C or D.

**Phases E and F** do not change. `_commit_tuning` (§4.2) replaces the inline commit at `:461-472`, and `_space` / `_used_default_space` come from the plan as before.

### 2.3 Resolution by the study's `space_mode` (5.2-1)

Generalise `resolve_config_space` (`_tuning_plan.py:83-135`) into:

```python
def resolve_space(space_mode: Literal["merge","replace"], user: Sequence[SearchDim],
                  provider: EstimatorProvider, task: TaskType) -> ResolvedSpace
def resolve_config_space(cfg, provider) -> ResolvedSpace:   # wrapper, same result as today
    return resolve_space(cfg.tuning.optuna.space_mode, parse_space(cfg.tuning.optuna.space), provider, cfg.task)
```

- The body is today's `:95-135` with `cfg` replaced by its arguments.
- `ResolvedSpace.fixed` is ignored for `space=` rounds, which use `study.fixed` (`:329`).
- The mode is the **study's** (`lineage.study.space_mode`). C13a already proved the config's mode equal to it, so `space_mode` cannot change on resume (5.2-1).

**Sharp edge, stated in the docs and pinned by a test.** In merge mode the declaration is the provider defaults with `space=` laid over them (5.2-1). A dim the *config* declared as user-written is not carried into a `space=` round unless it is repeated in `space=`:
- it falls back to the provider's default range;
- the 5.2-2 comparison then sees a changed declaration, so the dim's expansion is reset.

For example, `fast_config(mode="merge")` uses `SMALL_TREES = {3, 5}` for `n_estimators` (`tests/test_tuning/_resume_helpers.py:46`). A `space=` without `n_estimators` reverts it to 600–2500. Merge-mode PR 2 tests must therefore include `SMALL_TREES` in `space=`.

```python
@dataclass(frozen=True)
class Declaration:              # _tuning_space.py
    space_arg: tuple[SearchDim, ...]     # parse_space(space), raw names, as written
    declared: tuple[SearchDim, ...]      # resolved, names normalised to routes, H-0121 applied
    provenance: tuple[Provenance, ...]   # "user" for space= dims, "default" for provider dims
    routes_added: tuple[Route, ...]      # identities not routed before

def resolve_declaration(space_dims, *, space_mode, provider, task,
                        routes: Mapping[str, Route], base_smart: dict | None) -> Declaration:
    admit_space_routes(space_dims, routes, provider)        # §2.5, before anything else
    resolved = resolve_space(space_mode, normalise_names(space_dims, routes, provider), provider, task)
    declared = resolved.declared
    if base_smart is not None:                              # step 5 of §2.2
        declared, dropped = admit_smart_dimensions(provider, list(declared), base_smart, resolved.default_dims)
        log "tune.drop_dims" as plan_first_round does (_tuning_plan.py:273-277)
    routes_added = tuple(r for r in route_entries(declared, provider) if r.name not in routes)
    return Declaration(tuple(space_dims), tuple(declared), provenance kept for declared, routes_added)
```

- **H-0121 on every `space=` round** (5.2-4 last sentence; criterion 29 bullet 3). A merge `space=` re-adds the provider's `num_leaves_ratio` default, so it must be dropped again when `auto_num_leaves` cannot be true.
- A dim that `space=` declares is user provenance: it is refused (H-0121 decision 3), never dropped. It is kept when `space=` also declares an `auto_num_leaves` smart dim that can be true. That is the reading of "利用者が `space=` で宣言した同じ名前の次元は外れない".
- Provenance depends only on `space=` and the provider, so a reattach resolves the same thing (INV-R5).

### 2.4 Alias normalisation (D-b; criterion 29 bullet 5)

```python
def normalise_names(dims, routes, provider) -> list[SearchDim]
```

- Key each dim by `identity(d) = (d.category, canonical(d.name) if model else d.name)`.
- If that identity already has a route under another raw name, replace the dim's `name` with the routed raw name, via `dataclasses.replace(d, name=route.name)`.
- `space_arg` keeps the spelling the user wrote.

Why rename:
- `detect_boundary` looks up `best_params.get(dim.name)` (`search_space.py:386`), and Optuna keys params by raw name. A new spelling would make TPE treat the dim as a brand-new parameter, and expansion would miss the best value.
- After renaming, the route stays single (`routes_added` empty for that identity) and expansion is inherited by name.
- U2 compares by canonical name (§3), so alias and canonical spellings agree.

### 2.5 §4.1b category change (criterion 12)

```python
def admit_space_routes(space_dims, routes, provider) -> None
```

- Build `route_entries(space_dims, provider)` (`_tuning_plan.py:138-159`, raw names, model names canonicalised).
- Call `_admit_routes(routes, those)` (`_tuning_plan.py:162-174`) → `merge_routes` (`lineage.py:200-226`). That raises `RouteConflict` when:
  - the same raw name has another category; or
  - the same canonical name is reached from another category.
- The resulting `CONFIG_INVALID` uses `surface="tune(space=)"`. Give `_admit_routes` a `surface` kwarg; its message is already "a dimension's category cannot change within a study".
- **This runs first**, before the name checks. Otherwise `num_leaves_ratio` declared as `category: model` would get `check_param_names`'s "is a LizyML smart parameter" message, and `early_stopping_rounds` declared as model would get the training-managed message, instead of the 4.1b one.

### 2.6 Retargeted 5.2-4 admission (criterion 29 bullet 2)

`check_space_names(provider, cfg, *, space=None, surface="tuning.optuna.space")` keeps the body of `_model_tuning.py:528-548`, with three changes:
- `model_space_names(cfg, space=space, surface=surface)` reads the given raw dict instead of `cfg.tuning.optuna.space` (`_model_factories.py:634-650`).
- `check_duplicate_space_dimensions(provider, cfg, space=space, surface=surface)` runs on the **raw** `space=` dict, before merge resolution. `resolve_space`'s `overrides = {identity(d): d}` (`_tuning_plan.py:116`) would otherwise collapse `eta` + `learning_rate` silently.
- `check_training_managed_space(provider, cfg, space=..., surface=...)`.
- `check_calibration_param_names` runs only in the `space=None` call.

`validate_tuning_dimensions` (undeclared smart names) already runs on `plan.effective` in `check_round_space`, so it covers `space=` dims without change.

### 2.7 Record changes (`lineage_records.py`; schema stays 1)

- `RoundSource = Literal["config", "previous", "space_arg"]` (`lineage_codec.py:84-85`).
- `StoredRound.space_arg: tuple[SearchDim, ...] | None` (D-a).
  - Encoded as `encode_dims(space_arg)` in `"declaration.space_arg"` (`:659-661`) and decoded with `decode_dims` (`:688-690`).
  - The raw dict cannot be stored: `to_attr_value` uses `json.dumps(allow_nan=False)` (`:125`), so a NaN choice would turn an admissible `space=` into a `_codec_refusal` and break criterion 8b.
  - PR 1 only ever wrote `null`, and nothing is released (pr1-design.md:193), so no migration is needed.
- **New decoder rule:** `(source == "space_arg") == (space_arg is not None)`; anything else raises `LineageCodecError`, so a round record that breaks it reads as X.
- `RoundPlan` gains `source: RoundSource` and `space_arg: tuple[SearchDim, ...] | None = None`. `round_record()` (`_tuning_plan.py:222-239`) uses them, replacing the `round_number == 1` heuristic at `:226`.
- `provenance` for a `space=` round comes from `Declaration.provenance`.
- `routes_added = Declaration.routes_added`, filled with identities that are actually new, not `()` as at `:408`.

### 2.8 Planning a new round (`plan_next_round`; 5.2-2, 5.2-3, 5.3)

```python
def plan_next_round(cfg, provider, *, lineage, n_study_trials, best_params, best_value,
                    n_trials, expand_boundary, boundary_threshold,
                    declaration: Declaration | None) -> RoundPlan
    study, last = lineage.study, lineage.last_round
    if declaration is None:                                   # unchanged PR 1 path
        declared, prov, base, changed = last.declared, last.provenance, last.effective, frozenset()
        source, space_arg, routes_added = "previous", None, ()
    else:
        refuse_unchangeable(declaration.declared, lineage.rounds, lineage.routes)     # 5.2-3
        base, changed = carry_forward(declaration.declared, last,
                                      provider.parameter_bounds(cfg.task))        # 5.2-2
        declared, prov = declaration.declared, declaration.provenance
        source, space_arg, routes_added = "space_arg", declaration.space_arg, declaration.routes_added
    used_default = _uses_default(study.space_mode, prov)       # 5.3 bullet 4 / H-0102, per round
    should = expand_boundary if expand_boundary is not None else used_default
    if should and best_params is not None:
        frozen = changed | outside_range(base, best_params)    # 5.3 bullets 1 and 2
        effective, report = expand_for_next_round(base, best_params, threshold, frozen_names=frozen)
    else:
        effective, report = base, None
```

- **`carry_forward(declared, previous, bounds) -> (base, changed_names)`**
  - Key the dims by identity, `(category, routes[name].canonical)`; names are already normalised.
  - Equal declarations: same identity, and `dim_signature(d) == dim_signature(previous_declared)` (JSON of `encode_dim`, which is typed and NaN-aware). The dim inherits **previous.effective**, including its expansion and bounds.
  - Changed or new dims: `attach_bounds([d], bounds)`; the name goes into `changed`.
  - A dim missing from `declared` is removed. Later trials use the config value (6.1), and routes stay (4.1b).
  - `base` keeps the declared order.
- **`refuse_unchangeable(declared, rounds, routes)`**
  - `history` holds, for each identity, the latest declared dim from any prior round, including removed dims. The reason: Optuna checks compatibility against the first distribution it ever saw under a param name, and a dim can be removed and re-added.
  - Refuse with `CONFIG_INVALID` when any of these differ from the history dim:
    - the dim class (float / int / categorical);
    - `log`;
    - for categoricals, `tuple(encode_scalar(c) for c in choices)`. This compares order, type and value, keeps `True`, `1` and `1.0` apart, and makes NaN equal NaN.
  - Message: "…cannot change within a study; start a new study". Context: `{"surface": "tune(space=)", "dim", "changed": "kind"|"log"|"choices", "recorded", "given"}` (the recorded and given values as `repr`, so the context stays JSON-safe).
  - `low` / `high` changes are allowed (5.2-3 last bullet).
  - Note: `-0.0` vs `0.0` encode differently, while Optuna treats them as equal. This is stricter in the safe direction, so accept it.
- **`outside_range(base, best_params) -> frozenset[str]`**: numeric dims whose `best_params[d.name]` exists and lies outside `[d.low, d.high]`.
  - This can only fire in the round *after* a range change. Round k is frozen by bullet 1; round k+1 repeats the declaration, inherits round k's effective range, and the study best may be an older trial outside it. So criterion 30.1's out-of-range case needs a 3-round test.
- `expand_for_next_round(..., frozen_names=)` already exists (`_tuning_plan.py:308-359`). Frozen dims are reported with `expanded=False`, `new_*=None` and their `edge`, so `boundary_table` shows them.
- `plan_round` (moved dispatch):
  - C / F → `plan_next_round(declaration=resolve_declaration(...) if space_dims else None)`.
  - O → `plan_open_round(space_declaration=...)` (§3).
  - first → `plan_first_round`. Never with `space_dims`, because C-s1 refused that already.

---

## 3. U2: `space=` on an open round (7.2)

```python
def plan_open_round(cfg, *, lineage, trials, n_trials, expand_boundary, boundary_threshold,
                    space_declaration: Declaration | None) -> RoundPlan     # _tuning_plan.py:430
    ... existing n_trials / expand_boundary checks (:449-454)
    if space_declaration is not None and not same_declared_space(space_declaration.declared, record.declared, lineage.routes):
        _refuse_changed_argument(k, "space", <canonical summary of recorded>, <summary of given>)
```

- The open round's `Declaration` comes from the same `resolve_declaration` (§2.3), with `routes=lineage.routes`, which include the open round's own `routes_added`.
  - Its resolution includes the H-0121 drop. Without that, a dropped default dim would make "the same `space=`" compare unequal.
  - Route conflicts and parse errors raise `CONFIG_INVALID` before the comparison, which is the same code as a U2 refusal.
- **`same_declared_space(a, b, routes) -> bool`**
  - Set equality of `{(d.category, canonical(d)): dim_signature(replace(d, name=canonical(d)))}`. Canonical names come from the routes, falling back to the dim name.
  - The order of the dims in the dict does not matter. An omitted `log` is already `False` after `parse_space`, so the omitted and explicit spellings compare equal (8b).
  - `encode_dim` tags scalars, so a categorical whose choices differ only in type is unequal, and NaN equals NaN (8b).
  - Alias and canonical spellings compare equal (29 bullet 5).
  - `record.declared` carries no `min_allowed` / `max_allowed`, and neither does `Declaration.declared`, so the signatures are comparable.
- `_refuse_changed_argument` context values must be JSON-safe for `test_no_empty_context`. Pass a sorted list of `"category:canonical"` strings, not dims.
- The rule applies only to open rounds. After a closed round, `space=` starts a new round (§2.8). This follows the existing `test_after_closed_round_n_trials_free` pattern.

---

## 4. `read_study_lineage` (7.6) and `restore_tuning` (7.7)

### 4.1 `lizyml/tuning/study_lineage.py`

```python
StudyState = Literal["absent", "legacy", "uninitialized", "unreadable", "no_round",
                     "inconsistent", "open", "completed", "failed"]

@dataclass(frozen=True)
class RoundRecord:
    round: int
    status: Literal["completed", "failed", "open"]
    n_trials: int                 # requested (record budget.n_trials)
    n_finished: int               # COMPLETE + FAIL + PRUNED in the round's number range; RUNNING excluded
    best_score_before: float | None
    best_score_after: float | None   # None for failed and open

@dataclass(frozen=True)
class StudyLineage:
    state: StudyState
    rounds: tuple[RoundRecord, ...]   # () unless open / completed / failed
    open_round: int | None

def lineage_view(lineage: Lineage | None, trials: Sequence[TrialLike]) -> StudyLineage   # pure; None -> absent
def read_study_lineage(storage: str | BaseStorage, study_name: str) -> StudyLineage
```

- **`read_study_lineage`**:
  - `precheck_study_args(storage, study_name)`, now in `tuner.py`. Both `None` → `CONFIG_INVALID` ("pass storage and study_name").
  - `require_optuna()`.
  - `study = load_stored_study(storage, study_name, seed=0)`. Loading writes nothing (`tuner.py:79-94`).
  - `None` → `lineage_view(None, ())`.
  - Otherwise `lineage_view(classify_study(study.user_attrs, study.trials), study.trials)`.
- **State mapping.** The eight `LineageState` values (`lineage.py:93-102`) map one-to-one, plus `"absent"`. `classify_study` already applies the §7.4 order, including S before N (criterion 33).
- **`rounds`.** One record per entry of `lineage.rounds` / `lineage.ends` (`lineage.py:152-158`):
  - `status` is `"open"` when the end is `None`, otherwise `end.status`.
  - `n_finished` is counted from `trials_of_round` (`lineage.py:406-418`).
  - `failed` rounds are included, unlike `TuningResult.rounds`.
- **No writes.** No attribute writes, no RUNNING→FAIL, no identity check. The only storage side effect is the R7 schema creation of `get_all_study_names`. State that in the docstring and the docs.
- **Exports.** `lizyml/tuning/__init__.py` `__all__` gains `"RoundRecord"`, `"StudyLineage"` and `"read_study_lineage"`. `RoundRecord` is a documented component of the return type, so it must be importable by name. `lizyml.__all__` is unchanged (`tests/test_core/test_public_exports.py:15-31`).

### 4.2 `Model.restore_tuning`

```python
def restore_tuning(self, storage: str | BaseStorage, study_name: str,
                   data: pd.DataFrame | None = None) -> TuningResult       # on ModelTuningMixin
```

**Steps.** No study write anywhere; the only Model writes are in the final commit.
1. **A.** `validate_tune_args(cfg, boundary_threshold=0.05, storage, study_name)`: tuning present and `precheck_study_args`. Then `storage is None` → `CONFIG_INVALID`.
2. `require_optuna()`, **before any data load or storage contact**. The extras probe (§9 D-e) needs `OPTIONAL_DEP_MISSING` to come first.
3. `direction = resolve_tuning_direction(cfg)`.
4. **B.** `raw = self._load_data(data)`; `summary = summarise_frame(raw)`; `X, y, *_ = self._prepare_training_data(raw)`; `provider = get_provider(cfg.model)`. These are the same per-call transients as in tune (pr1-design R8).
5. **C.**
   - `study = load_stored_study(storage, study_name, seed)`. `None` → `TUNING_FAILED` ("no stored study to restore").
   - `lineage = classify_study(...)`, then `refuse_state(...)`: L / S / X → `CONFIG_INVALID`.
   - `open` → `CONFIG_INVALID`: "round k is open; finish it with `tune(resume=True)`", context `{"surface": "tuning.study", "state": "open", "round": k}`.
   - `uninitialized`, `no_round` or `failed` → `TUNING_FAILED`, context `{"state": ..., "study_name": ...}`.
   - Then `check_study_direction`.
   - Then `identity, versions, _ = objective_identity(cfg, summary, provider, direction)` and `check_against_record(...)`. This covers the §5.1 config space, the §4.3 identity (`CONFIG_INVALID`) and the version warning, which continues.
6. **D (pure).** `restored = restore_from_study(lineage, study.trials, study.best_trial, metric_name, direction)`:

   ```python
   @dataclass(frozen=True)
   class Restored:  # _tuning_restore.py
       result: TuningResult; round_number: int; effective: tuple[SearchDim, ...]
       used_default: bool; fixed: Mapping[str, Any]
   # result = assemble_tuning_result(lineage, trials, best.params, best.value,
   #            metric_name=..., direction=..., current_round=lineage.last_round.round)
   # effective = last.effective; used_default = _uses_default(study.space_mode, last.provenance)
   # fixed = lineage.study.fixed
   ```

   - It returns exactly what the closing tune returned. tune builds its result with the same function at `_model_tuning.py:449-457` and `current_round` = the last round, so the result and the record's `boundary_report` (with its own threshold) are reused rather than recomputed.
7. **Commit.** `self._commit_tuning(X=X, y=y, result=..., study=study, round_number=..., effective=..., used_default=..., fixed=..., storage=storage, study_name=study_name)`.
   - `_tuning_fixed_params = study.fixed`, so the next `fit()` trains as the best trial did (INV-R2).
   - The remembered pair is the arguments, so a later `tune(resume=True)` continues this study as round k+1.
   - `_fit_result` / `_refit_result` / `params_table` inputs are not touched (H-0094 decision 13).
   - No run dir is created, and `_provider` is not set (it is transient).

```python
def _commit_tuning(self, *, X, y, result, study, round_number, effective,
                   used_default, fixed, storage, study_name) -> None   # replaces :463-472
```

tune calls it with `storage` / `study_name` = the held pair (`:461-462` logic kept at the call site).

**Pre-H-0122 pickled Models** are handled already: `getattr` defaults at `:257-258`, and the commit sets both fields.

---

## 5. Reporting surfaces (6.5, criterion 16a)

**State today.**
- PR 1 already takes `TrialResult.round` from the lineage (`rounds.py:100`, `round_of`).
- `tuning_table` already writes `t.round` (`_model_tables.py:229`), so it needs only a test.
- The plot still sums `RoundSummary.n_trials` and only draws when `len(rounds) > 1` (`plots/tuning.py:89-118`). After a `failed` round 1, `rounds == (round 2,)`, yet the trials span two rounds, so no separator is drawn.

**New rule** (`plots/tuning.py`):

```python
def _round_spans(trials) -> list[tuple[int, int, int]]:   # (round, first_number, last_number)
    # sort by number; group consecutive trials with equal .round
if len(spans) > 1:
    for _, first, _ in spans[1:]: fig.add_vline(x=first - 0.5, ...)
    by_round = {rs.round: rs for rs in tuning_result.rounds}
    for k, first, last in spans:
        label = f"Round {k}" + (f" ({', '.join(rs.expanded_dims)})" if (rs := by_round.get(k)) and rs.expanded_dims else "")
        fig.add_annotation(x=(first + last) / 2, ...)
```

- A round with no `RoundSummary` (a failed round) gets a plain label. A `TuningResult` built by a user may also lack summaries, so a "(failed)" suffix could be wrong there.
- After `Model.load`, `trials == []` (`_model_persistence.py:499`), so the plot has no spans, the same as today.

**Existing pins.** `test_retune.py:1114-1201` (`test_single_round_no_vlines`, `test_two_rounds_has_vline`) stay green, because their trial rounds agree with their summaries. No existing test pins the cumulative computation specifically. The new tests are in §8.

---

## 6. Fix for #342 (expansion narrows or inverts a range)

**Root cause.** The clamps in `_expand_range` and `_expand_int_range` (`search_space.py:319-325`, `:355-362`) can push an edge *inward* past the original:
- `min_allowed > low`, which the #342 case hits;
- the linear `0.0` floor at `:313` when `low < 0`;
- the `IntDim` `max(1, …)` guard at `:362` when `low < 1`: `max_depth` with min −1, `bagging_freq` with min 0. It narrows the low edge even when only the **upper** edge expands.

**Fix.** Expansion never narrows. After the clamps and guards:

```python
# _expand_range, after :319-325
new_low = min(new_low, low); new_high = max(new_high, high)
# _expand_int_range: compute lo = max(1, floor(new_low)), hi = ceil(new_high); then
return min(lo, low), max(hi, high), clamped
```

- **Effect on #342.** `IntDim n_estimators [3, 5]` with `min_allowed=10` and the best at 3: the clamp to 10 is undone to 3, so the range is unchanged. H-0111's existing check (`:429`, `:443`) then reports `expanded=False`, `new_*=None`, `clamped_to_bound=True`. This is the H-0111 clamp no-op report.
- **Defensive invariant in `expand_dims`** (`:563-596`). Emit `low=min(dim.low, new_low)` and `high=max(dim.high, new_high)`, and raise `ValueError` if the result has `low >= high`. LizyML's own paths cannot reach the raise after the fix.
- **Unchanged pins.** `test_int_dim_lower_guard` (`test_retune.py:94-106`, low 2 → 1, which is outward), `TestNoOpExpansionIsNotExpanded` (`:355-470`) and `test_one_side_still_moves_is_expanded` all still hold.
- **Why it lands in PR 2.** `_resume_helpers.SMALL_TREES = {3, 5}` sits below `n_estimators` min 10, so every merge-mode resume with `expand_boundary=True` and the best at 3 already hits #342. Criterion 29 bullet 4 (a user range outside the provider bound, then expansion) is the same scenario.

**Tests: `tests/test_tuning/test_expansion_never_narrows.py`**
- **Unit**, parametrized cases:
  - int below `min_allowed` (the issue's exact repro);
  - float below `min_allowed`;
  - float above `max_allowed` at the upper edge;
  - int `low=0` with an upper-edge expansion (the guard must not move low);
  - float `low=-1` at a linear lower edge (the floor must not move low);
  - `max_depth`-style `IntDim(-1, 5)` at the lower edge.
- Each case asserts `new_low <= low`, `new_high >= high`, no `low > high` from `expand_dims`, and the expected `expanded` / `clamped_to_bound`.
- **Through `Model.tune`** (`test_user_range_below_bound_is_not_inverted`):
  - `fast_config(mode="merge")`, round 1, then `resume=True, expand_boundary=True`.
  - Use a spy (or a 1-value range trick) to make round 1's best `n_estimators == 3`: declare `n_estimators` categorical-free int `[3, 4]` with `threshold=0.49`, so either value is an edge.
  - Assert:
    - the round-2 effective `n_estimators` range is still `[3, 4]`;
    - the round-2 trials are COMPLETE, not FAIL;
    - the `boundary_table` row has `expanded=False` and `clamped_to_bound=True`.
  - **Bug-first check:** `git stash` the fix and the test fails.
- **Docs.** One bullet in BLUEPRINT §11.5 非対称拡張ルール (§7). A CHANGELOG `Fixed` bullet that `Fixes #342`. HISTORY: H-0122 実装の記録 2 (§9 D-f).

---

## 7. Documentation

### 7.1 Edits

**BLUEPRINT.md** (Japanese).
- **§11.3:**
  - Rewrite `:887-892` ("Resume retains … overlay") to the H-0122 rules: a round's trials train on `config ⊕ fixed`; `fixed` comes from the study; the space comes from the round record or `space=`.
  - Rewrite the H-0121 range paragraph at `:909`. It names `load_if_exists=True`, so restate it as "a tune that creates a study (`resume=False`, U3, state E/N)". Keep the token `get_all_study_names`, which is an H-0121 anchor (`docs/proposal_dispositions.toml:500`).
- **§11.4** TuneProgressInfo `current_trial` row (`:961`): "この round の何番目（途中の round の再開では続きから）" (C11).
- **§11.5:**
  - Add `space` to the parameter table (`:999-1006`).
  - Add the #342 bullet to 非対称拡張ルール (`:1014-1022`): "拡張は範囲を狭めも反転もしない…".
  - Update the `rounds` comment at `:1054-1058`: failed rounds are omitted, so `rounds` can start at round 2.
  - Update `plot_tuning_history` (`:1076-1080`): boundaries come from trial rounds.
  - Replace 制約 (`:1093-1097`).
- **§11.5.1:** replace `:1110` (`load_if_exists=True`), resume との関係 (`:1112-1116`) and 制約 (`:1118-1122`). The H-0099 bullet must lose "before enqueue".
- **New sections:**
  - §11.5.2 `space=` (5.1–5.3).
  - §11.5.3 study records and the §7.4 state table.
  - §11.5.4 `read_study_lineage` / `StudyLineage` / `RoundRecord`: the arguments, all nine `state` values, and that it only reads.
  - §11.5.5 `restore_tuning`.

**docs/api.md.**
- `tune` (`:66-111`):
  - Add `space` to the signature and the table.
  - Rewrite the paragraph at `:86-90` without enqueue or warm-start.
  - Rewrite the `resume` row (`:96`): reattach with `resume=True`, and create the study when the storage lacks it (U3).
  - Fix the `study_name` row (`:101`, "re-using the same pair re-attaches").
  - Replace Raises (`:105-109`) with every §8 condition.
- New `#### restore_tuning` after `tune`.
- `TuningResult.rounds` row (`:401`): remove "Empty tuple for single-round tuning" and add "a round closed `failed` is omitted, so `rounds` can start at round 2; the next round's `best_score_before` is `None`". Also fix `boundary_report` (`:402`).
- `TuneProgressInfo.current_trial` (`:456`).
- New `## Study lineage (lizyml.tuning)`, placed before `## Error Codes`. It holds `read_study_lineage`, a `### StudyLineage` table and a `### RoundRecord` table, each followed by `---` (needed by the `_api_fields` parser).
- The `TUNING_FAILED` row in the error table (`:497`): add "`restore_tuning` with no closed round to restore".

**CHANGELOG.md `[Unreleased]`.** PR 1 added nothing, so all of H-0122 goes here.
- `### Added`: `tune(space=...)`, `lizyml.tuning.read_study_lineage` / `StudyLineage` / `RoundRecord`, `Model.restore_tuning`.
- `### Changed -- action may be required`:
  - the identity check;
  - `resume=False` onto an existing study is refused;
  - reattach takes `resume=True`, and U3 creates the study;
  - resume rounds no longer re-evaluate the previous best, so results may differ (10.1);
  - the new error conditions;
  - U1: a study created before v0.19.0 cannot be continued; finish paused jobs before upgrading.
- `### Fixed`: #342; round separators after a failed or timed-out round; a removed smart dim no longer reaches LightGBM (6.4); `current_trial` above `n_trials` on reattach (P4b).
- Fix `:383`, the old enqueue bullet, only if CHANGELOG history is editable; recommend leaving released sections untouched.

**docs/migration.md.** New `## v0.19.0` above `:8`. It holds the U1 paragraph, a "Calls that now raise `CONFIG_INVALID`" table (§8 rows → what to write instead), "Results that change" (no enqueue), and "Changing the space on resume: `tune(space=...)`".

**HISTORY.md H-0122.**
- Set ステータス to Accepted.
- Append `### 実装の記録 2: PR 2`, in Japanese, recording D-a … D-f and the R7 resolution.

**docs/proposal_dispositions.toml.** H-0122 becomes `disposition = "specified"` with `anchors = ["read_study_lineage", "StudyLineage", "RoundRecord", "restore_tuning"]`. These must be full tokens in both BLUEPRINT and the H-0122 HISTORY entry; all four already occur in the HISTORY text at 7.6 / 7.7. The `[names]` table is unchanged.

**Optional.** The tutorial notebook text near its resume cell mentions enqueue (pr1-design §7). No test asserts the text.

### 7.2 Doc tests

**New file `tests/test_docs/test_resume_docs.py`.** It follows `test_embargo_merge_docs.py:17-23` (`_read`, `_section`, substring asserts). The doc edits must use these exact anchors:
- `api_tune = _section(api, "#### `tune`", "#### `restore_tuning`")`.
  - Contains `space: dict[str, Any] | None = None` and `` | `space` | ``.
  - Contains none of `enqueue`, `warm-start`, `re-attaches to the persisted study`.
  - Contains `` `resume=True` `` together with `creates the study`.
  - Contains each §8 anchor: `` `study_name` without `storage` ``, `already holds`, `objective`, `created before`, `cannot be read`, `disagree`, `` `space_mode` ``, `distribution`, `category`, `recorded`, `` `space=` ``, `` `TUNING_FAILED` ``.
- `api.md` `### TuningResult` section: the `rounds` row contains `failed` and `round 2`, and not `Empty tuple`.
- `api.md` `#### `restore_tuning`` and `## Study lineage` sections:
  - contain `read_study_lineage`, `StudyLineage`, `RoundRecord`;
  - contain every `StudyState` value, derived from `typing.get_args(StudyState)`, not a list;
  - contain `read-only`.
- BLUEPRINT sections `## 11.5` up to `## 11.6`:
  - contain `space=`, `resume=True`, `照合`, `read_study_lineage`, `restore_tuning`, every `StudyState` value, and `読み取りだけ`;
  - contain neither `load_if_exists=True` nor `完全後方互換`;
  - contain no `enqueue` except in the phrase `enqueue しない`. Assert `text.count("enqueue") == text.count("enqueue しない")`.
- CHANGELOG `_section(text, "## [Unreleased]", "## [0.18.0]")` and migration `_section(text, "## v0.19.0", "## v0.18.0")` must both contain:
  - `created before v0.19.0`
  - `before upgrading`
  - `` `tune(space=...)` ``
  - `` `resume=True` ``
  - `CONFIG_INVALID`
  - `previous best` (the no-enqueue result change)

**`tests/test_docs/test_result_field_inventories.py:52-68`:** add `CASES` entries `(StudyLineage, _api_fields("### StudyLineage"))` and `(RoundRecord, _api_fields("### RoundRecord"))`.

`test_proposal_blueprint_coverage.py` picks up the new disposition automatically.

---

## 8. Test plan

Helpers are reused from `tests/test_tuning/_resume_helpers.py`: `fast_config`, `frame`, `url`, `load`, `snapshot_study`, `snapshot_model`, `die_in_objective`, `die_at_attr`, `crash_round_2`, `spy_tuner`, `delete_attrs`, `set_attr`, `INCONSISTENT`, `SMALL_TREES`.

**Helper additions:**
- `die_after_attr(monkeypatch, key)`: writes, then raises `ProcessDeath`.
- `space_of(cfg)`: a deep copy of the config space, as a base for `space=`.
- `round_record(storage_or_study, k)`: decodes `lizyml:round:k`.
- `spy_fit_params(monkeypatch, model)`: records `_build_train_components` kwargs.

### 8.1 Unit tests (pure)

- **`tests/test_tuning/test_space_declaration.py`:**
  - `resolve_space` merge/replace parity with the old `resolve_config_space`.
  - Merge `space={}` → defaults, `used_default`.
  - `normalise_names` alias → routed name.
  - `admit_space_routes` conflicts (raw name, canonical).
  - `refuse_unchangeable`: kind, log, choices order, `True`/`1`, `1`/`1.0`, NaN equals NaN, removed-then-re-added dim.
  - `carry_forward`: unchanged → previous effective; changed/new → bounded new range plus name in `changed`; removed → absent.
  - `outside_range`.
  - `same_declared_space`: dict order, omitted `log`, alias, choice type, NaN.
- **`tests/test_tuning/test_lineage_records.py`** (extend):
  - a `space_arg` round trip with NaN / `True` / `1.0` choices;
  - `source="space_arg"` with dims accepted;
  - `source="previous"` with dims refused.
  - `:203` stays green, now for the iff reason; add a comment.
  - `_lineage_fixtures.stored_round` (`:97-100`) already takes `**overrides`, so it needs no change.
- **`tests/test_tuning/test_study_lineage_view.py`:** `lineage_view` over synthetic lineages: every state, `n_finished` excludes RUNNING, `failed` rounds included.
- **`tests/test_tuning/test_tuning_exports.py`** (golden, new): `set(lizyml.tuning.__all__)` is pinned to the old 7 names + `RoundRecord`, `StudyLineage`, `read_study_lineage`. `lizyml.__all__` stays unchanged (existing test).

### 8.2 Criterion map

| # | file::test | Mechanics |
|---|---|---|
| 4 | `test_resume_space_arg.py::test_moved_range_samples_new_range_and_keeps_old_best` | Replace mode, round 1 `learning_rate` [0.05, 0.3]. Round 2 `space=` with `learning_rate` [1e-4, 2e-4] log (other dims as before) and `expand_boundary=True`. Round-2 trial values are in the new range. `best_params["learning_rate"]` is the round-1 value. `spy_fit_params` shows `fit` uses it. The `boundary_table` `learning_rate` row has `expanded=False`. |
| 6 | `::test_removed_smart_dim_routes_to_smart` | `auto_num_leaves=True`; space has `num_leaves_ratio` (smart) and `learning_rate`. Round 2 `space=` without the ratio and with a bad `learning_rate` range keeps the round-1 best. The round-2 spy `smart_params["num_leaves_ratio"]` is the config value. `best_smart_params` has the ratio. `tests/_train_spy.record_lightgbm_calls` during `fit` shows no `num_leaves_ratio` key. Replaces the deferred flow noted at `test_space_merge.py:320-323`; that comment is updated to point here. |
| 6b | `::test_removed_training_dim_stays_training` | Same, with `early_stopping_rounds` (training): `best_training_params` has it, `best_model_params` does not, and `fit`'s `training_overrides` carries it. |
| 7 | `::test_unchangeable_refused[kind, log, choices_true_to_1, choices_1_to_1p0, choices_order]` | Built on the C study template. Each case gives `CONFIG_INVALID` with `context["changed"]`, and `snapshot_study` / `snapshot_model` are unchanged. `space_mode` stays covered by PR 1's `test_resume_identity.py::test_config_space_change_refused[space_mode]`; add `[space_mode_with_space_arg]` (config mode edited + `space=`) → still §5.1. |
| 8b | `test_resume_open_round.py::test_u2_space_same_passes[same, reordered_omitted_log, alias]`, `::test_u2_space_different_refused[range, choice_type]`, `::test_u2_space_nan_choice_resumes` | Round 2 is started with `space=S`, `die_in_objective`, then a new Model reattaches with the variants. Refusals leave the RUNNING trial RUNNING and the attributes unchanged. NaN case: `S` has `lambda_l1` categorical `[0.0, nan]`; the reattach with the same `S` passes and round 2 closes `completed` (trials sampling NaN may FAIL; the round-1 COMPLETE trials keep it completed). |
| 12 (category) | `test_resume_space_arg.py::test_category_change_refused[smart_to_model, training_to_model]` | `num_leaves_ratio` as `category: model`, and `early_stopping_rounds` as model. Expect `CONFIG_INVALID` with surface `tune(space=)` and `"category"` in the message (4.1b message, not the name check). Study unchanged. |
| 14 (space) | `test_resume_call_matrix.py::test_space_on_create_refused[memory, storage_absent]`, `::test_space_on_u3_refused` | `resume=False` + `space=`: no storage → Model unchanged; storage with a new name → `not Path(db).exists()`. U3 → `name not in optuna.get_all_study_names(url)`, and no `tune.study_created` log in caplog. |
| 15 (space) | `test_resume_space_arg.py::test_space_round_has_no_enqueue` | As PR 1 (`test_resume_lineage.py`): no `"fixed_params"` in `system_attrs`, round count == `n_trials`, best == study best. |
| 16a | `tests/test_plots/test_tuning_plot_rounds.py::test_separators_follow_trial_rounds[failed_round1, timeout_round]`, `::test_cumulative_summaries_disagree_unit`, `::test_tuning_table_round_column` | (a) The 8a flow (`n_trials=1`, die in the only trial, reattach → `TUNING_FAILED`, `resume=True` → round 2). `fig.layout.shapes` x == `[first_number_of_round_2 - 0.5]`; annotations "Round 1", "Round 2" at the span midpoints. (b) Timeout 0.5 s with `n_trials=1000` in round 1, round 2 normal: vline x equals the first round-2 trial number − 0.5. (c) Synthetic `TuningResult` whose `RoundSummary.n_trials` disagree with the trial rounds: the vline follows the trials. Fails under the old cumulative code. (d) `tuning_table()["round"].tolist() == [t.round for t in result.trials]`, and equals `round_of` from the decoded records. |
| 17 | `test_resume_space_arg.py::test_expand_none_after_user_dim_same_after_reattach` | Merge mode with `SMALL_TREES`. Round 2 `space=` adds the user dim `max_depth`. Variant A: same Model, round 3 `expand_boundary=None`. Variant B: a copy of the storage, new Model reattach, round 3. Both round-3 records have `expand_boundary is False` and equal `boundary_report`. |
| 19 (space) | `test_resume_lineage.py::test_space_on_no_round_refused`, `::test_space_on_uninitialised_refused` | `die_at_attr(round_key(1))` → N; `die_at_attr(STUDY_KEY)` → E. `resume=True, space=…` → `CONFIG_INVALID`; no `lizyml:study` (E) or round record (N) written. |
| 22 (PR 2 rows) | `test_resume_refusal_invariance.py` | Add `Case.method: Literal["tune","restore_tuning"] = "tune"` and `Case.args: tuple = ()`. The test dispatches `getattr(case.model, case.method)(*case.args, **case.kwargs)`. Rows: `space_on_create[memory]`, `space_on_create[storage_absent]` (`extra={"no_file": url}`), `space_on_u3`, `space_on_no_round`, `space_on_uninitialised`, `space_kind`, `space_log`, `space_choice_type`, `space_choice_order`, `space_category[smart_to_model]`, `space_unknown_name`, `space_alias_duplicate`, `space_training_as_model`, `space_inactive_smart_user`, `space_malformed_shape`, `open_space[range]`, `open_space[choice_type]`, `restore[open]`, `restore[failed]`, `restore[uninitialised]`, `restore[no_round]`, `restore[absent]` (Model-only snapshot), `restore[legacy]`, `restore[unreadable]`, `restore[inconsistent:gap]`, `restore[identity_data]`. |
| 29.1 | `test_resume_space_arg.py::test_merge_space_changes_one_default_keeps_others` | Merge mode; `space=` holds `SMALL_TREES` + `learning_rate` changed. Round-2 trials sample every default dim (by param keys). Record provenance: `learning_rate` and `n_estimators` user, the rest default. |
| 29.1 (sharp side) | `::test_merge_space_omitting_config_user_dim_reverts_to_default` | `space=` without `n_estimators` → the round-2 declared `n_estimators` is the provider default; it is listed as changed (no expansion). Use `n_trials=1` and a tiny frame; the round is slow but bounded. Alternatively assert on the planned record via `die_at_attr(end_key(2))` before trials. |
| 29.2 | `::test_admission_refused[unknown_name, alias_duplicate, training_as_model]` | Each → `CONFIG_INVALID` before trials, study unchanged. |
| 29.3 | `::test_h0121_drop_on_space_rounds[in_process, reattach]` | Merge mode, `auto_num_leaves=False`. Round 1 drops `num_leaves_ratio` (PR 1). Round 2 `space=` (merge) → declared lacks the ratio; `tune.drop_dims` logged. Same through reattach. Plus `space=` with `num_leaves_ratio` and `auto_num_leaves` smart categorical `[True, False]` → the ratio is kept. |
| 29.4 | `::test_out_of_bound_user_range_not_clamped` | `space=` with `n_estimators` [3, 5] (min_allowed 10). Trials sample 3–5; round 3 `expand_boundary=True` never goes past bounds or inverts (shares the #342 Model test). |
| 29.5 | `::test_alias_then_canonical_one_route_inherits` | Replace mode, round 1 `eta` [0.05, 0.3]; round 2 `expand_boundary=True` expands `eta`; round 3 `space={"learning_rate": [0.05, 0.3]}` (same declaration). Exactly one route with canonical `learning_rate`; the round-3 effective `eta` equals round 2's expanded range; trial param keys stay `eta`. The open-round alias pass is in 8b `[alias]`. |
| 30.1 | `::test_changed_dim_near_edge_not_expanded`, `::test_best_outside_range_not_expanded_next_round` | (a) Round 2 `space=` moves `learning_rate` to [best·0.99, best·3] with `expand_boundary=True` → the `boundary_table` row has `edge=="lower"` and `expanded is False`. (b) Three rounds: round 2 moves the range away (frozen), round 3 has no `space=` and `expand_boundary=True`; the round-1 best is outside → `expanded is False`. |
| 34 | `tests/test_docs/test_resume_docs.py` (§7.2), `test_result_field_inventories.py` (2 cases) | |
| 35 | `tests/test_tuning/test_read_study_lineage.py::test_state[absent, legacy, uninitialized, unreadable, no_round, inconsistent, open, completed, failed]`, `::test_reads_only`, `::test_precheck[storage_only, name_only, neither]`, `::test_frozen_and_tuple`, `::test_exports` | Builders: a fresh URL; `delete_attrs("lizyml:%")`; `die_at_attr(STUDY_KEY)`; `set_attr(... {"schema": 999})`; `die_at_attr(round_key(1))`; `INCONSISTENT["gap"]`; `crash_round_2` with storage (RUNNING trial); two rounds; the 8a flow. Expected `rounds` / `open_round` per state (failed round 1 included). `snapshot_study` before and after are equal, and the RUNNING trial is still RUNNING. `FrozenInstanceError` on assignment to both dataclasses. |
| 36 | `tests/test_tuning/test_restore_tuning.py::test_restore_equals_live_result`, `::test_crash_after_end_then_restore_then_round3`, `::test_fit_after_restore_uses_best_and_fixed`, `::test_restore_writes_nothing`, `::test_refusals[...]`, `::test_restore_on_fitted_model_keeps_fit`, `::test_version_difference_warns`, `::test_threshold_report_reused` | (a) Run two rounds to `live`, then a new Model `restore_tuning(url, "s")`. Compare `best_*_params`, `best_score`, `rounds`, `[(t.number, t.round, t.state)]` and `boundary_report` field by field (FAIL scores are NaN). (b) `die_after_attr(end_key(2))` → state C → restore → `tune(resume=True)` writes `lizyml:round:3`. (c) Merge mode: the `fit` spy shows `model_params` ⊇ `best_model_params` and `first_metric_only is True`. (d) `snapshot_study` is equal and the round count is unchanged. (e) Refusals as in the criterion-22 rows. (f) After `fit()`: `_fit_result` / `_refit_result` keep their identity and `params_table()` is equal, while `_tuning_result` changes. (g) `set_attr` versions → `pytest.warns(UserWarning)`. (h) Round 1 `lambda_l1` FloatDim [0, 1] in replace mode, best position `p` (deterministic seed; assert `min(p, 1-p) < 0.4`). Round 2 `boundary_threshold = min(p, 1-p) + 0.01`. Restored `boundary_report == live` and `!= detect_boundary(prev_effective, best, 0.05)`; for a lower edge the row is `edge=="lower"`, `expanded is False` (0.0 floor), which is the spec's [0, 1] / 0.08 / 0.10 case. |
| #342 | §6 | |

### 8.3 Existing tests to update (`dir(Model)`-derived populations and registries)

- `tests/test_persistence/_load_census.py:174`: add `"restore_tuning": "restores tune state from a stored study; a lifecycle step, not a reading"`. `test_reporting_surfaces_survive_load.py:41-44` asserts `set(INVENTORY) == public`.
- `tests/test_calibration/test_calibration_param_names.py:165`: `NON_TRAINING_ENTRY_POINTS["restore_tuning"] = "reads a study's records and trials; trains nothing"`.
- `lizyml/_extras.py:79` row → update `tests/test_docs/test_extras_registry_table.py:56-62` (expected set) and add `CallForm("restore_tuning", _tuning, {}, ("sqlite:///<tmp>/s.db", "s"))` to `tests/test_notebooks/probe_extras_registry.py:105` (D-e).
- `tests/test_tuning/test_space_merge.py:320-323`: the comment now points to `test_resume_space_arg.py`.
- `tests/test_tuning/test_resume_lineage.py:235-256` (frozen hook test): unchanged; it still pins `expand_for_next_round`.

### 8.4 TDD order for the makers

Each step should be RED first, then GREEN:
1. #342 fix (unit test, then the Model test).
2. Codec split plus `space_arg`.
3. `_tuning_space` unit functions.
4. `_tuning_admission` move, then `_model_factories` kwargs.
5. Planner: `plan_round`, `plan_next_round(declaration)`, U2.
6. `tune(space=)` wiring for phases A/C/D; criteria 4, 6, 6b, 7, 12, 14, 15, 17, 19, 29, 30.1, 8b.
7. `study_lineage` and the exports (35).
8. `_tuning_restore` and `restore_tuning` (36), then the census, calibration and extras updates.
9. Plot (16a).
10. Criterion-22 rows.
11. Docs and doc tests (20, 34), HISTORY, dispositions.
12. Check every touched file against the line cap (`wc -l lizyml/**/*.py`, each ≤ 800).

---

## 9. Risks, open questions and decisions

- **D-a. `space_arg` is stored as encoded dims, not as the raw dict.** The raw dict cannot carry a NaN choice through `to_attr_value` (`lineage_codec.py:116-128`), and 8b requires that round to work. The parsed form is still "the declaration as written" (raw names and order, with omitted keys filled in). **Recommended**: record this in 実装の記録 2. No schema bump (pr1-design.md:193).
- **D-b. An alias is renamed to the routed raw name** (§2.4). The alternative, the new spelling as a second route, breaks expansion inheritance and makes TPE see a new parameter. **Recommended**: rename, and record it.
- **D-c. R7.** `resume=False` + `space=` is refused in phase A, so nothing touches storage, which meets criterion 14 literally. U3 / E / N + `space=` needs `get_all_study_names`, which may create the SQLite schema; the study is still not created and no attribute is written. **Recommended**: accept this and state it in 実装の記録 2 and in the `read_study_lineage` docstring, which has the same side effect. No maintainer action unless "no file at all" is required for U3, which the spec's U3 row does not demand.
- **D-d. `restore_tuning` stays on `ModelTuningMixin`.** H-0091 and `test_mixin_state_isolation.py:198-216` name one writer mixin.
- **D-e. Extras registry row for `restore_tuning` — maintainer decision.** `test_extras_registry_table.py:54-64` pins the set to "the table H-0119 states". Adding `("restore_tuning", None, "tuning")` changes that statement. Without it, the registry under-reports a public method that needs optuna. **Recommended**: add the row and the probe call form, and record it in 実装の記録 2 as an amendment-by-extension of H-0119's table. `restore_tuning` calls `require_optuna()` before loading data or contacting storage.
- **D-f. Where #342 is recorded — maintainer decision.** Option 1 is H-0122 実装の記録 2, since the fix lands in PR 2 and changes §11.5. Option 2 is a separate HISTORY entry. **Recommended**: option 1, plus the CHANGELOG `Fixed` bullet.
- **R-1. Merge `space=` re-defaults config user dims** (§2.3). This follows spec 5.2-1 literally and may surprise users. Docs state it, and a test pins it. No spec change.
- **R-2. The provider default space may change across library versions.** The space is not part of the identity (4.2). A merge `space=` round resolves defaults from the current provider, so a changed default shows up as a changed declaration and its expansion is reset. Rounds without `space=` inherit the record and are unaffected. Accept and document.
- **R-3. 5.2-3 history scope.** Comparing against every prior round's dims, including removed ones, is stricter than "the previous round" and matches Optuna's own first-distribution check. **Recommended** as designed.
- **R-4. `-0.0` vs `0.0` choices** are treated as different (tagged JSON), while Optuna treats them as equal. This only errs toward refusing (INV-R4-safe), so accept it.
- **R-5. `restore_tuning(None, None)`.** precheck passes when both are `None`. **Recommended**: `CONFIG_INVALID` ("restore_tuning reads a stored study"). The spec's TUNING_FAILED list covers stored-study states, not missing arguments. `read_study_lineage(None, None)` behaves the same way.
- **R-6. `restore_tuning` refusal order.** Absent → `TUNING_FAILED` comes first, then L/S/X → `CONFIG_INVALID`, then O → `CONFIG_INVALID`, then E/N/F → `TUNING_FAILED`, then direction and identity (`CONFIG_INVALID`) for C only. Changed data on an F study therefore gives `TUNING_FAILED`. The spec lists the cases but not their precedence; this order means "nothing to restore" wins over "would mismatch". **Recommended** as designed.
- **R-7. Bounds lookup under an alias.** `attach_bounds` matches by `dim.name` (`search_space.py:500`), so an `eta` dim gets no `learning_rate` bounds. This already exists in round 1 and PR 2 does not change it; mention it in 実装の記録 2 as out of scope (no issue filed here).
- **R-8. Config `space` shape is not validated.** `config/schema.py:556` types it as `dict[str, Any]`. PR 2 validates only `space=` (A-s1); the config gap is out of scope.
- **R-9. Line budget.** If `_model_tuning.py` ends above ~760, move `_commit_tuning` into `_tuning_restore.py` as a free function taking the Model. It contains no `overlay_params`, so the seam scan is unaffected. Do not move `_build_tune_objective`.
- **R-10. The plot label for a failed round** is a plain "Round k" (§5). Ask the maintainer if a "(failed)" suffix is wanted; it would be wrong for user-built `TuningResult`s without summaries.
- **R-11. Flaky determinism.** Criteria 4, 6 and 6b rely on round 2 not beating round 1. Make the round-2 range clearly worse, e.g. `learning_rate` 1e-4 with 3 trees, and assert the best trial's number is below round 2's `start_trial` as an explicit precondition. Criterion 36(h) asserts its precondition on the seeded position instead of skipping.

### Critical Files for Implementation
- /home/rem/repos/LizyML/lizyml/core/_model_tuning.py
- /home/rem/repos/LizyML/lizyml/core/_tuning_plan.py
- /home/rem/repos/LizyML/lizyml/tuning/lineage_codec.py (split into lineage_records.py)
- /home/rem/repos/LizyML/lizyml/tuning/search_space.py
- /home/rem/repos/LizyML/lizyml/plots/tuning.py
- New: /home/rem/repos/LizyML/lizyml/core/_tuning_space.py, /home/rem/repos/LizyML/lizyml/core/_tuning_admission.py, /home/rem/repos/LizyML/lizyml/core/_tuning_restore.py, /home/rem/repos/LizyML/lizyml/tuning/study_lineage.py
- Tests: /home/rem/repos/LizyML/tests/test_tuning/_resume_helpers.py, /home/rem/repos/LizyML/tests/test_tuning/test_resume_refusal_invariance.py, /home/rem/repos/LizyML/tests/test_persistence/_load_census.py, /home/rem/repos/LizyML/tests/test_docs/test_extras_registry_table.py, /home/rem/repos/LizyML/tests/test_core/test_fit_params_override.py (SEAM_MODULES constraint)