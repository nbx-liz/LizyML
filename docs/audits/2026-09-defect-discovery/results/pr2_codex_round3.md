# PR 2 — Codex review, round 3 (2026-09-07)

Rounds 1 and 2 are in the sibling files. Before this round a **relational**
monitor comparing rounds 1-2 returned `CONVERGING` / `continue`
(`results/pr2_monitor_round12.md`), on the ground that each round closed its
axis by derivation from an authority rather than by adding the missed instance.

---

## Verdict

```
VERDICT: REQUEST_CHANGES
```

Two blocking findings. The first is the declared deliverable failing for the
first time — not a remedy's remedy, but `fit(params=)` not reaching the model.

### 1 — an alias override loses to a lower-priority canonical name

`lizyml/core/model.py`. `{**base, **override}` merges by **spelling**. LightGBM
resolves aliases and, when both spellings reach it, uses the canonical one. So
an override spelled as an alias survived into the dict and was then ignored:

```
{'learning_rate': 0.5} ['[learning_rate: 0.5]']
{'eta': 0.5}           ['[learning_rate: 0.001]']
```

**Defect-class: DC2**, presenting as DC1.

### 2 — the config's collision refusal still admits aliases

`lizyml/config/schema.py`. `LGBMConfig(params={"max_leaves": 12},
auto_num_leaves=True)` is accepted where `num_leaves` is refused, and trains at
32. Same literal-comparison shape as round 2's finding, on the config surface.

---

## Disposition

### Finding 1 — fixed, and it was larger than the report

Measuring it separated two things the first probe could not. `_COMMON_DEFAULTS`
injects **`learning_rate=0.001` into every fit**, so the canonical spelling is
present whether the user wrote it or not. Values read from the trained booster:

| config | `fit(params=)` | before | after |
|---|---|---|---|
| `learning_rate: 0.07` | — | 0.07 | 0.07 |
| **`eta: 0.07`** | — | **0.001** | **0.07** |
| `learning_rate: 0.07` | `eta: 0.5` | **0.07** | **0.5** |
| (none) | `eta: 0.5` | **0.001** | **0.5** |

The second row is the one the reviewer's framing missed: this was never only a
`fit(params=)` defect. **A config setting any of the eleven defaulted
parameters under an alias has been inert since it shipped.**

So the fix is at two seams, and both are needed — fixing only the facade leaves
the canonical default to be re-injected downstream and the override still loses,
which was measured rather than assumed:

- `overlay_params(provider, base, overlay)` merges by identity and is used at
  each seam in `_merge_params` (config → provider fixed → tune best → override).
- `LGBMAdapter._build_params` drops a default whose parameter the user has named
  under any spelling.

**The estimator is never handed two spellings of one parameter**, so the outcome
does not depend on which one it prefers. That preference was measured, but the
fix does not rest on it.

`overlay_params` deliberately **does not drop a name the estimator does not
know**: dropping it would hide the name from H-0093's refusal and turn a typo
back into a silent no-op. Pinned by a test.

A public Protocol method `canonical_param_names` carries the identity, since
`core/model.py` must stay estimator-agnostic. `BLUEPRINT.md` §14.4 lists it.

RED verified at both seams: reverting the facade overlay reddens the alias
override cells; removing the adapter's shadow-drop reddens five cells including
the config case. The CHANGELOG records the behaviour change, because a config
that was silently training with a default will now train with its own value.

### Finding 2 — half already fixed, half deferred and filed

**The DC5 half was fixed before this round's verdict arrived**, at `31a25a6`
(pushed 2026-09-07, after the reviewer had read the tree): `BLUEPRINT.md` §5.3
and H-0094 now say the config surface matches **literal names only**, with the
`max_leaves` / `min_child_samples` measurements. The reviewer read the earlier
text, which did describe those collisions as refused. The record is corrected
rather than argued with.

**The DC2 half is #280**, filed from the same measurement before the verdict
arrived. It is not fixed here: `lizyml/config/schema.py` cannot import
`estimators/` under the 5-layer DAG, so the alias table is unreachable from the
layer the check lives in, and where to move the refusal is a design decision
rather than a detail. Measured `0/824` of the configs the shipped suite builds
hit it.

That scope call is recorded rather than assumed to be right. If round 4 holds it
blocking, it goes to the maintainer as a decision, not as a scope change made
under review pressure.

---

## Checked and clean (round 3)

- 53 passed on the override file (59 after this remedy).
- **Registry correspondence**: every entry compared against
  `lightgbm.basic._ConfigAliases._get_all_param_aliases()` — **140 canonicals /
  307 spellings**, exact agreement in both directions.
  `accepted_spellings("max_leaves")` and an unknown name both raised.
- **Other name gates**: 66 passed; source inspection confirmed the search-space,
  calibration and export accepted-name checks all share the alias-inclusive
  registry. One export test could not set up — the sandbox had no writable
  temporary directory.
- **The deferral**: §5.3's tuning-space exclusion and the 54/67 bound remain
  explicit.
- DC1–DC7 reviewed; no further reproduced blocker.

## State

Blocking findings per round: **1, 1, 2**. Full suite **2252 passed**;
`ruff check .`, `ruff format --check .`, `mypy lizyml/` clean.
