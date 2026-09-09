# PR 2c acceptance criteria — H-0098

Base: develop at 2436a66c268ba80a205460220d114cdaf662cbaf.
Status: local implementation; external review and maintainer acceptance pending.

## Contract

Issue #284: one structural walk derives normalization, unchanged status and
mapping presence. Both predicates consume those facts; no second recursive
definition of the accepted set remains. Scalar and element conversion continue
to preserve their distinct estimator wire representations.

Issue #287: retain the deliberate H-0095 difference. Categorical choices accept
only exact plain scalars; numeric bounds become plain at parse time. Numpy
categorical choices must fail at the named entrance, never as a domain failure
inside a study. Widening the choices set is not part of this change.

## Evidence map

| Requirement | Evidence |
|---|---|
| Wire preservation and unchanged accepted set | Existing `tests/test_core/test_param_domain.py` population, 5263 passed / 230 skipped in focused execution |
| Mapping exclusion and caller-independent predicates | Same population, identity and numpy print-option adversarial cases |
| One structural dispatch | `_walk` in `lizyml/core/param_domain.py`; `_holds_a_mapping` and `_is_unchanged` removed from production |
| Generated numpy/position contract remains synchronized | `instruments/param_domain_contract.py --check` passed |
| Every literal-bearing dimension | `tests/test_tuning/test_literal_domain_contract.py`, categorical/float/int cases |
| Actual numpy rejection and successful plain tuning | `instruments/space_choice_normalisation.py`, executed at base: CONFIG_INVALID versus successful eta=0.5 |

The original #287 symptom is historical and already fixed at this base. This PR
records and tests the remaining policy decision; it must not claim to newly fix
the old TUNING_FAILED behavior. Issue closure requires separate tracker approval.

## Review boundary

Review the exact committed diff, H-0098, this table and final check results.
No new provider review invocation has been authorized for PR 2c. PR 2b's
conditional repair-only capacity cannot be transferred. No merge is authorized.
Any finding that changes the accepted domain requires a fresh explicit contract
decision rather than an incidental refactor adjustment.
