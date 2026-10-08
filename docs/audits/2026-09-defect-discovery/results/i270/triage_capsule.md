# Task capsule — #270 hollow-test triage (read-only)

**Objective.** For every test in your input file, decide which category below it belongs to, with evidence. You answer for the classification file only.

**Background.** An instrument classified each test's *claim* by matching nouns in its name + docstring ("training", "fit", "predict", "metric", "fold", ...) to a producer set, then ran the test with that producer set patched to raise (`ProducerRan`). Every test in your input **still passed** with the producers named in `producers_killed` disabled. So it is a measured fact that each test never executes those producers. The open question is whether that is a defect: the noun matcher is crude (it reads "fit" in `test_fit_result_...`, "prediction" in a metric unit test, "training" in "refused before training").

**Categories (pick exactly one).**

- `HOLLOW` — read honestly, the test's name/docstring claims an effect that only the killed producer can produce (e.g. "param reaches the booster", "fit(params=...) overrides", "the trained model differs", "OOF covers every row" through the real pipeline), and the test observes a value one step upstream (a helper's dict, a private merge function, a hand-built FitResult). Give a one-line repair: what boundary the test must reach and what to assert there.
- `NEGATIVE` — the claim is that something is refused / does not happen / happens *before* the producer runs. Passing with the producer disabled is exactly the claim. Confirm the test asserts the refusal (raise, error code) and does not merely pass vacuously.
- `UNIT` — the claim, read honestly, is about the unit the test actually calls (e.g. "WAPE of a perfect prediction is zero" calling the WAPE function; "isotonic calibrator maps extreme scores to extreme probabilities" calling the calibrator). The noun matched, but the claim does not need the killed producer. Name the unit and confirm the test calls it.
- `OVERCLAIM` — the name/docstring claims a boundary effect, the body only checks an upstream value, **and** another test already asserts that effect at the boundary. Name that sibling as `path::name` and confirm by reading it that it reaches the boundary. Fix = narrow this test's name/docstring. If no such sibling exists, it is `HOLLOW`, not `OVERCLAIM`.
- `UNCLEAR` — you cannot decide from the source; say what would decide it.

**Rules.** Read each test's source (and its fixtures/helpers when needed) at the current working tree `/home/rem/repos/LizyML` (branch `develop`, head `1d41b66`). Judge from the name + docstring + body, not from the category you would like. Do not run the full test suite (CPU is shared). You may run single tests if needed. When torn between `UNIT` and `HOLLOW`, ask: would a reader of the test name believe the library's end-to-end behaviour is covered? If yes and it is not, `HOLLOW`.

**Writer / mutation surface.** Read-only for the repository. The only file you may write is your output file (path given in your prompt).

**Output.** A JSON array, one object per input test, in input order:
`{"test": "<as in input>", "category": "...", "claim": "<the claim in <=15 words>", "observes": "<what the body actually asserts, <=15 words>", "evidence": "<path:line>", "repair": "<HOLLOW: one-line repair; OVERCLAIM: sibling path::name; else empty>"}`

Then reply with: counts per category, and the 5 HOLLOW items you consider most important (the ones a filed issue or a real defect class lives behind), one line each. Keep the reply under 300 words; the file holds the detail.

**Stop conditions.** Every input test has exactly one object in the output file. Do not modify any repository file. Do not propose fixes beyond the one-line `repair`.
