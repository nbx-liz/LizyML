# H-0130 amendment 2 — confirmation review checklist (fixed before the review)

This file fixes, before the review runs, what the one confirmation review of the revised amendment 2 checks (the maintainer's decision 3-A, 2026-10-15). The review answers each item and nothing else.

Subject: the section `### 改訂 2（2026-10-15、管理者の判断で受け入れ）` (titled `### 改訂 2（2026-10-15、提案。管理者の判断を待つ）` while under review) at the end of `HISTORY.md`, revised after the reviews of its first draft (`6338c73`).

## Items to check

1. **The decisions are stated as decided.**
   - 1-B: no registry check of any kind; exception 2 is written as the registry helpers' role.
   - 2-A: selection by Config type, discriminator `name` or a derived value is part of 組み立て.
   - 3-A: this review.

   For each, are the declared bounds stated, and does the amendment avoid claiming more than R0-R5 check? R0-R5 check static import edges only.
2. **The removal list is complete.**
   - Change 2's "消すもの" covers every file, symbol, test and negative control that depends on R6 (`_dispatch_rules.py`, `dispatch_allowed`, `DispatchKey`, `DISPATCH_ALLOWED`, `dispatch_problems`, `R6_PASSES`, the two negative controls).
   - Its line references are correct at the reviewed head.
   - The helpers the probes need (`_package`, `_bindings`, `_dotted`, `_Index`) have a stated new home.
3. **The factual corrections from the first draft's reviews are applied.**
   - The two reviews are `h0130-amend2-review` and the fresh-context review.
   - The corrections:
     - 77 of 78;
     - the seventeen imports sentence;
     - `calibration/registry.py:44-52`;
     - the H-0054 rows labelled as past;
     - "判断 c" removed;
     - the O5 compatibility decision in H-0131;
     - the end condition of the review-only period.
   - The `h0130-combo-critique` wording is adopted for exception 2, 2-A and the status text.
4. **The baseline and the two new should-fail cases are specified so they can be executed.**
   - R0-R5 and the tables stay unchanged.
   - Adding `from lizyml.config.schema import LGBMConfig` to `evaluation/evaluator.py` gives R3.
   - Adding the same line to `estimators/provider.py` gives R2.
5. **One scope, one end condition, everywhere.** The review-only scope and its end condition ("until H-0131's implementation and the matching architecture tests are merged") are identical in:
   - the BLUEPRINT text (change 1);
   - the PR template item (change 2);
   - the CHANGELOG sentence (change 3);
   - the PR-body requirement (change 3);
   - acceptance criterion A2-5.
6. **H-0131 carries over what this amendment leaves open.**
   - O4, O5 and 1-C are listed as candidates, none as decided.
   - Item 9 keeps selection by name or value separate from the import-based options.
   - Item 10 states the end condition.

Every item is answered by reading or running code at the reviewed head. The probes and `tests/test_architecture/` train nothing.

## Out of scope for this review

- Choosing a different option (O1-O9, 1-A, 1-C, ii) or reopening the maintainer's decisions 1-B, 2-A and 3-A.
- Designing H-0131 beyond checking that its scope list carries the items above.
- Reviewing the implementation. Nothing in change 1-5 is implemented until the maintainer accepts the amendment.
- New wording preferences that do not make a statement false or inconsistent.

## Verdict rule

`APPROVE` when items 1-6 hold, with no blocking or major finding. A finding outside items 1-6 is reported as information and does not by itself change the verdict, unless it shows a statement in the amendment to be false.
