**RECOMMENDATION: C**

| Option | False COMPLETE possible? | Real-row over-refusal | Loop closed by construction? | Effort/risk |
|---|---|---|---|---|
| A | Yes; excluded dynamic paths still reproduce it | 0/19 diagnostic rows | Closes the two demonstrated gaps, not overall import reachability | Small patch; residual boundary risk |
| B | Yes; both known failures remain | 0 observed | No; documenting exceptions does not prevent them | Least effort; retains DC1 |
| C | Staging-induced false RED eliminated; mutation/test declarations remain fallible | #277 needs migration; other 16 before-tree rows retain RED | Yes, for the staging hazard | Remove staging/guard; add one mutation declaration and focused review |
| D | Existing tool remains vulnerable if used | Unchanged | No | No immediate work; delivery delayed |

**Measurements**

- Verified checkout `1f155e29893184a7447ac90073843c7246692a74` and local `origin/develop` at `33a3f6e`.
- [Reproduced both round-6 counterexamples](/tmp/pr8c-round6-probes-90attzzu/results.json): original before **1 passed**, staged before **1 failed**, after **1 passed**, instrument **COMPLETE**.
- Applied A’s rules to a scratch copy: both counterexamples become **INCOMPLETE**; existing positive controls remain detected. Runtime-name and cross-file examples still produce false COMPLETE under the declared exclusions. [Results](/tmp/pr8c-option-a-probes-xw31irz2/results.json).
- Ran `pr8c_round2_staging.py` through a read-only adapter using locally verified merge commits and existing worktrees. A produces **zero references**, with output identical to the saved staging measurement. [Replay](/tmp/pr8c-options-v7o7b28u/staging_a.txt).
- Corrected denominator: **19 diagnostic rows / 13 before commits includes two mutation rows**. Actual before-tree execution covers **17 rows / 12 commits**. Sixteen diagnostic rows have new-file candidates; fourteen belong to before-tree execution.
- [Executed all 17 before-tree rows without new package files](/tmp/pr8c-options-v7o7b28u/no-staging-results.json): **16 retain failed test nodes without collection errors**. Only **#277** fails collection, missing `_optimizer.py`. Restoring staging yields **102 failed, 27 passed**.
- Inspected #296’s fixing diff: discarding configured parameters in either Platt or Beta is a plausible reintroduction mutation. Both candidate assignments occur once in the after tree and were added by #296. Neither mutation was implemented or validated.
- Repository unchanged; scratch writes stayed under `/tmp`; all instrument/pytest runs used the three required thread limits.

**Reasoning**

C’s measured migration cost is **one additional row**, not fourteen or nineteen. Most copied modules are unnecessary for obtaining RED. Removing package staging eliminates the mechanism that manufactured these false failures and ends this particular open-grammar review loop.

A is defensible as bounded risk acceptance: it fixes both demonstrated gaps without observed over-refusal. It does not establish general safety. B retains a concrete DC1 failure: reaching an inspection limit silently permits certification. For C, DC7 is manageable: sixteen rows demonstrably remain satisfiable, while #277 needs an executed, reviewed mutation before migration is complete.

**What the maintainer should know before choosing**

- C changes #277’s evidence to **COMPLETE-RED-BY-MUTATION**, which must remain distinct from historical before-tree RED.
- Review the mutation’s semantic relevance, provenance, collection success, and intended failing assertions; “some test failed” is insufficient.
- C closes the staging hazard, not every limit of completion certification.
- This is a recommendation, not acceptance or authorization to merge.