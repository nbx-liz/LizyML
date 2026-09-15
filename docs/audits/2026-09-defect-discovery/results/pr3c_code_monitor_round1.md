# PR 3c code review — loop monitor after round 1 (absolute, 2026-09-15)

Carrier: Codex gpt-6-astra, effort low, read-only, fresh context. Capsule: `prompt-templates/pr3c-code-monitor-round1.md`.

VERDICT: DELIVERABLE-FOCUSED

### Classification

- **Finding 1 — deliverable change plus supporting apparatus.** The string check changes runtime refusal behavior at [`_optimizer.py:125`](/home/rem/repos/LizyML/lizyml/calibration/_optimizer.py:125). The shared malformed-method cases and fit/tune assertions are **apparatus** directly supporting that change: `test_calibration_param_contract.py`’s refusal lists and `test_calibration_params_reach.py`’s refusal tests.

- **Finding 2 — deliverable change plus supporting apparatus.** The approximately 70 template lines become executable validation and warning handling in the exported, LizyML-independent `train.py`; they change what the delivered program accepts. See [`templates.py:563`](/home/rem/repos/LizyML/lizyml/codegen/templates.py:563), `_check_cal_params`, `_run_minimize`, and their fitter calls. This directly serves H-0100 decision 4’s runtime/generated optimizer-contract parity. The generated-module fixture, JSON-representability filter, and shared refusal-case test are **apparatus** supporting that parity. The test covers the shared cases expressible in JSON, excluding tuple-only cases.

- **Finding 3 — apparatus: issue tracking.** Filing the convergence-handling concern adds no implementation to this deliverable. It follows the bounded disposition described in acceptance criteria §4 for nonblocking B3 findings.

- **B4 evidence gaps — apparatus.** Additional refusal coverage associated with rows 6a, 6c, and 8b supports the two production remedies. Filing the remaining rows as one issue is tracking work. These actions do not add a new harness, workflow, or review gate. This classification does not certify that the tests close every listed evidence gap.

### Proportion

The remedy set is proportionate to the declared value: parameters must be honoured or refused across runtime and exported consumers. Generated code is expressly part of the deliverable in [H-0100](/home/rem/repos/LizyML/HISTORY.md:10028), design §3.6, and BLUEPRINT §15.4.

Duplicating validation creates maintenance cost, but its size alone does not establish apparatus drift: the added code executes in the user’s exported program. The accompanying tests address the resulting parity obligation. Deferring convergence work and broader evidence improvements keeps this round bounded, consistent with §4’s rule that B4 work does not trigger re-review.

### Recommendation

**continue** — The proposed remedies remain directed at declared runtime and export behavior, with bounded supporting tests and deferred follow-ups.

## Main-context disposition: continue

Adopted. Round 2 is scoped to the two round-1 fixes only (acceptance criteria §4, B1: one limited verification of the fix).
