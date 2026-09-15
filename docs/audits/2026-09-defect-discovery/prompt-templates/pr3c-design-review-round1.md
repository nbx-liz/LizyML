Review-kind: review
Review-round: 1

# PR 3c design review — does the plan honour the original calibration design?

You are reviewing a **design**, not code. Nothing is implemented yet. You have
**read-only** access to the repository; change nothing, run no network calls
beyond reading the repository, and do not open pull requests or edit issues.

## Exact head

- Repository: `/home/rem/repos/LizyML`
- Branch `fix/phase3-pr3c-calibration-params`, branched from `develop` at `5ac725e`.
- The design under review:
  `docs/audits/2026-09-defect-discovery/results/pr3c_design.md` (Japanese).

## Background, stated as fact

Issue #277: `calibration.params` is accepted for every calibrator, but `platt`
and `beta` receive it and discard it. The maintainer decided to **honour** the
parameters rather than refuse them, and asked that the **original functional
design** be followed. The maintainer also fixed the overridable set for `beta`
to `x0`, `method`, `bounds`, `tol` and `options`.

## The question

**Does the design faithfully implement the original design of the calibration
surface, and does it avoid contradicting any higher-priority document?**

The repository's document priority is: `BLUEPRINT.md` > `HISTORY.md` >
`AGENTS.md` > `skills/` > code. Judge against those, not against the design's
own claims about them.

## Sources to check the design against (read them yourself)

- Commit `b465240` (`git show b465240`) — introduced `CalibrationConfig.params`
  and `get_calibrator(name, params=None)`.
- `BLUEPRINT.md` §12.1-12.4 (calibration), §15.4 (`export_code`), §18.1.2
  (test perspectives), and the layer rules in `ARCHITECTURE.md`.
- `HISTORY.md` entries **H-0030** (raw scores), **H-0031** (Beta), **H-0047**
  (Isotonic parameters), **H-0058** (outer split reuse), **H-0059**
  (`export_code`), **H-0090** (codegen reproduction), **H-0093** (parameter
  name gate, including `calibration.params`), **H-0094** decision 8
  (`calibration.params` canonicalisation), **H-0095** (value normalisation and
  the accepted set).
- `.claude/skills/calibration/SKILL.md`.
- Code: `lizyml/calibration/{platt,beta,isotonic,registry,base,cross_fit}.py`,
  `lizyml/codegen/{templates,config_writer,generator}.py`,
  `lizyml/core/_model_factories.py` (`check_calibration_param_names`,
  `canonicalise_calibration_params`), `lizyml/core/model.py` (calibration wiring),
  `lizyml/config/schema.py` (`CalibrationConfig`).

## What to check, specifically

1. **Traceability.** For each element of design sections 3.1-3.3, is it grounded
   in an original design source, or is it an addition? Name the source, or say
   it has none.
2. **Contradictions.** Does anything in the design contradict a higher-priority
   document — the Isotonic override pattern, OOF-only and cross-fit rules, raw
   score input, the Beta model form, the `export_params` / `predict` contract, the
   layer rule that `lizyml/calibration/` may not import `lizyml/estimators/`, or
   the placement reasons H-0093 gives for the name gate?
3. **Omissions.** Does the original design imply something the plan leaves out —
   a position where `calibration.params` should reach a consumer and the plan does
   not name it, a forced value the Isotonic pattern would suggest for Platt or
   Beta, a documentation obligation?
4. **Design section 5 open points.** Give a recommendation for each: forced values
   for Platt; whether Beta `bounds` (a list of three pairs) and `options` (a dict
   with string keys) fit the H-0095 accepted set; whether the facade is the right
   place for the name gate.
5. **The export_code reproduction scope.** The plan also fixes the generated
   code for `isotonic`, whose runtime path already honours the parameters. Is
   that required by the original design of `export_code`, or is it scope creep?

## Output

At most **1200 words**.

1. `VERDICT: ADHERES`, `VERDICT: ADHERES-WITH-CHANGES`, or `VERDICT: DEVIATES`.
2. **Findings**, most severe first. Each: severity (blocking / should-change /
   note), the design section, the source it conflicts with or is missing
   (file and section or line), and the concrete change you recommend.
3. **Answers to the section 5 open points.**
4. **Bounds**: what you read, what you did not read, and what you did not verify.

Do not implement anything and do not grade code that does not exist yet.
