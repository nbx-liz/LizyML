# Mechanized sweeps of BLUEPRINT's own enumerations

`mechanize-on-recurrence`: a documented gotcha that recurs must be promoted from
prose to something that fires at action time. This one recurred twice by
reading — section 14.4's Protocol at 11 methods against a real 14, and
`BoundaryDimStatus` at 9 fields against a real 10 — and both were found by a
reviewer reading carefully, which is the retrieval path that had already failed
once. So the class was closed mechanically instead.

Three sweeps, each comparing something BLUEPRINT **presents as complete**
against its machine-readable counterpart.

## `r26_declared_blocks.py` — every printed class declaration

```
RoundSummary        BLUEPRINT:841-865    6 declared /  6 real   OK
BoundaryDimStatus   BLUEPRINT:841-865    9 declared / 10 real   SHORT  -> clamped_to_bound
BoundaryReport      BLUEPRINT:841-865    2 declared /  2 real   OK
EstimatorProvider   BLUEPRINT:1221-1244 11 declared / 14 real   SHORT  -> parameter_bounds,
                                                                          objective_choices,
                                                                          metric_choices
4 printed declarations checked, 2 differ.
```

**The class is closed at two instances.** Both were already in the item list, and
there is no third — a negative result worth having, because it stops the search.

The parser skips multi-line signature parameters (paren depth), which its first
version counted as attributes and reported as two phantom extras.

## `r27_config_tables.py` — section 5.4's nine Config tables

Section 5.4 calls itself 「全キー一覧」. One omission was found in it by hand;
the other eight sub-tables had never been swept.

```
トップレベル      10 keys / 11 fields   DIFFERS -> output_dir
data              4 /  4   OK
features          3 /  3   OK
model（LightGBM） 7 /  8   DIFFERS -> name
training          2 /  2   OK
tuning            1 /  1   OK
evaluation        1 /  1   OK
calibration       2 /  3   DIFFERS -> params
8 tables checked, 3 differ.
```

**Two of the three are new** and appear in none of the 62 clauses:

- **`model.name`** — `schema.py:271` declares `name: Literal["lgbm"]`, **required,
  no default**. It is the discriminated-union tag. A reader building a config
  from section 5.4's model table has no way to know it is needed; a config
  without it fails validation with `CONFIG_INVALID`. (Verified the hard way
  earlier in this session: a probe config written from the table's keys was
  rejected with `Unable to extract tag using discriminator 'name'`.)
- **`calibration.params`** — absent from the calibration table, while
  `BLUEPRINT.md:999` elsewhere says 「`calibration.params` で上記デフォルトを
  上書き可能」. The document uses the key it does not list.

`output_dir` was already carried as an item (H-0034 / H-0039).

## `r28_enum_sweeps.py` — the ErrorCode list and the metric lists

```
ErrorCode   section 16.2 lists 16 / enum has 20
            absent: EVALUATION_FAILED, CALIBRATION_NOT_FITTED,
                    TARGET_NOT_NUMERIC, TARGET_UNSEEN_LABEL
metrics     binary      registry 8, all 8 named, list open-ended
            multiclass  registry 6, all 6 named, list open-ended
```

**The ErrorCode gap is reported with a caveat that cuts against it:** section
16.2's heading is 「例外コード（例）」 — *examples*. An explicitly non-exhaustive
list is not obliged to be complete, so this is weaker than the section 5.4 case,
where the heading claims completeness.

It does bear on **#263**, though. That issue's evidence is that three members are
"documented" at `BLUEPRINT.md:1351/1356/1359` and unraised. If section 16.2 is a
list of examples rather than a register, "documented" is a weaker claim than #263
makes of it — and four *raised* members are missing from the same list, which
#263 does not mention.

The metric lists are complete for the tasks they cover and are explicitly
open-ended (`...`), so nothing is owed there — which retroactively supports the
H-0004 `SPECIFIED` verdict and the H-0071 `MISSING` one resting on `:1569`
rather than on section 13.1.

## What this demonstrates

Three rounds of careful reading did not find `model.name` or
`calibration.params`. Two sweeps, each under a hundred lines, found both in one
pass — and also established that the printed-declaration class has **no third
instance**, which reading could never have established.
