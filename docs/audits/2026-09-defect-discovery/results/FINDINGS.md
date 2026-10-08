# LizyML — defect discovery, findings register

- Head under discovery: `5712f41` (`main`, tree identical to `origin/develop`)
- Working tree: clean throughout (`git status --porcelain` = 0 lines)
- Plan: `discovery-plan.md` v12, APPROVED at review round 11

**What this file is.** A register of findings with stable ids, each carrying the
evidence that produced it, and a disposition table naming the artifact that
carries the work. It follows the convention `lizy-harness` uses in
`docs/deepseek-harness-review.md`: findings are grouped by mechanism rather than
one row per member, no Issue numbers are invented, and the register itself is
the written disposition for members that do not warrant their own Issue.

**What this file is not.** It is not a Proposal under CLAUDE.md section 2, and it
grants no implementation authority. Every repair listed here goes through the
ordinary Change Gate on its own merits.

---

## D2 — same-topic specification consistency matrix

**Population:** 162 anchors from six schemas — 94 config declarations, 24 `Model`
public members, 20 `ErrorCode` members, 13 stage entry points, 8 module-level
public constants, 3 task types. Each count reproduced by `d2.py` against the
plan's declared figure.

**Result:** 162 classified, **0 unclassified**. 57 `AGREE`, 90 `CANNOT-TELL`,
5 `CONTRADICT` rows (3 distinct findings), 2 `ONE-SIDED`, 8 `UNDOCUMENTED`.
Stop condition **REACHED**.

**Positive controls, both re-detected:** `FORMAT_VERSION` and
`TimeHoldoutInnerValid`, each `CONTRADICT`.

**Two instrument defects were found and fixed before any verdict was recorded**,
both of the classes this plan hunts:

- *Under-retrieval (DC1).* The first sweep matched anchors case-sensitively.
  `ARCHITECTURE.md:48` spells the constant `format_version`, so two of the three
  stale sites of this step's own control were invisible and the step would have
  reported no contradiction.
- *Over-reporting (DC2).* The first value comparison took every
  `anchor = literal` on a line as a claim about that anchor's default, and
  produced **8 false positives out of 9** `CONTRADICT` rows: `Field(exclude=True)`
  is pydantic's keyword, not `FeaturesConfig.exclude`; `enabled=False の場合` is a
  branch condition, not a default; `ratio: 0.25` is an example illustrating a bug.
  Occurrences inside another callable's argument list and inside conditional
  clauses are now rejected, and the three that still survived were adjudicated
  individually and recorded as `AGREE`.

### D2-F1 — `TimeHoldoutInnerValid` purge behaviour, stated both ways in BLUEPRINT

`BLUEPRINT.md` section 10.3.1 (L602) says the auto-resolved
`TimeHoldoutInnerValid` purges `purge_gap + embargo` (`purged_time_series`) or
`gap` (`time_series`) between inner-train and inner-valid, citing H-0085/#212.
`BLUEPRINT.md` section 10.3.3 (L646) says the class carries no purge or embargo
of its own even when the outer CV has one. Same document, same class, opposite
claims.

The implementation sides with 10.3.1: `training/inner_valid.py:166-178` declares
a `gap` parameter whose docstring cites H-0085/#212, and
`core/_model_factories.py:297` passes it on the auto-resolve path.

**Recommended resolution:** 10.3.1 is authoritative — it is the later decision
and the implemented one. 10.3.3 L646 is the pre-#212 description of the class
and should be rewritten to say the class accepts a resolver-supplied gap.

**Why this blocks other work:** K-07 was filed as an implementation defect and
retracted precisely because this clause disagrees with itself. D3's disposition
for `TimeHoldoutInnerValid.gap` depends on this decision.

### D2-F2 — shuffle against a time-ordered split: forbidden or warned?

`BLUEPRINT.md` section 8.2 (L537) lists `shuffle 禁止` among the time-series
validators — forbidden. `BLUEPRINT.md` section 10.3.1 (L599) permits a shuffling
`method: holdout` inner valid against a time-ordered outer split, emits a
`UserWarning`, and states explicitly that behaviour is unchanged and the user's
explicit choice is respected (H-0085/#210).

Forbidden against permitted-with-warning is a contradiction of modality, not a
difference of detail. Affects three config rows (`KFoldConfig.shuffle`,
`StratifiedGroupKFoldConfig.shuffle`, `GroupCVConfig.shuffle`).

**Recommended resolution:** 10.3.1 is authoritative as the later and more
specific decision; 8.2's entry should be restated as detect-and-warn, or scoped
explicitly to the outer split.

### D2-F3 — `ARCHITECTURE.md` states `format_version = 1`; the code ships 2

Three sites in `ARCHITECTURE.md` state version 1 — `:48` (layer diagram node),
`:487` (class diagram member `+FORMAT_VERSION = 1`), `:646` (artifact layout).
`persistence/exporter.py:38` ships `FORMAT_VERSION = 2`, `loader.py:35` accepts
`{1, 2}`, and `BLUEPRINT.md:1290` states 2. DC3: a derived document stale against
its source of truth, with nothing checking it.

**Recommended resolution:** update the three `ARCHITECTURE.md` sites, and add a
check that fails when a version stated in a document diverges from the constant.

### D2-F4 — `ARCHITECTURE.md` holds no rank in the document hierarchy

`CLAUDE.md` section 1 ranks BLUEPRINT > HISTORY > AGENTS > skills > code and does
not place `ARCHITECTURE.md` anywhere. It is an 814-line document describing the
layer DAG, and D2-F3 is a contradiction *inside* it that cannot be adjudicated by
the stated hierarchy — the rank has to be decided before the contents can be
judged.

For this pass `ARCHITECTURE.md` was treated as unranked, so its statements were
compared but never given authority over BLUEPRINT.

**Recommended resolution:** give it an explicit rank in `CLAUDE.md` section 1, or
state explicitly that it is a derived document with no authority — which is what
D2-F3 suggests it has become.

### D2-F5 — eight implemented names appear in no document

| Anchor | Where |
|---|---|
| `ErrorCode.EVALUATION_FAILED` | `core/exceptions.py` |
| `ErrorCode.CALIBRATION_NOT_FITTED` | `core/exceptions.py` |
| `SUPPORTED_CONFIG_VERSIONS` | `config/loader.py` |
| `TASK_TYPES` | `core/types/task.py` |
| `CHECKSUM_ALGORITHM` | `persistence/exporter.py` |
| `DEFAULT_TEMPLATE` | `plots/_theme.py` |
| `DEFAULT_HEIGHT` | `plots/_theme.py` |
| `DEFAULT_WIDTH` | `plots/_theme.py` |

`CHECKSUM_ALGORITHM` is the sharpest and corroborates D5's control from the
opposite direction: H-0083 decided a SHA-256 checksum per artifact, the code
implements it, and `BLUEPRINT.md` — the document CLAUDE.md section 3 requires to
fix the Artifacts contract — contains the word "checksum" zero times.

`ErrorCode` members that no document names are the mirror image of #263, where
three documented members are raised by nothing.

### D2 dispositions

| # | Action | Carrying artifact |
|---|---|---|
| 1 | Decide 10.3.1 vs 10.3.3 and rewrite the losing clause (D2-F1) | New Issue; blocks the D3 disposition for `TimeHoldoutInnerValid.gap` |
| 2 | Decide forbidden vs warned for shuffle against a time-ordered split (D2-F2) | Same Issue as 1 — one H-0085 cluster, one decision |
| 3 | Update the three stale `ARCHITECTURE.md` version sites and add a drift check (D2-F3) | New Issue, together with 4 |
| 4 | Give `ARCHITECTURE.md` a rank, or declare it derived and unauthoritative (D2-F4) | Same Issue as 3 |
| 5 | Document the eight names, or record them as deliberately internal (D2-F5) | New Issue; `CHECKSUM_ALGORITHM` is shared with D5 |
