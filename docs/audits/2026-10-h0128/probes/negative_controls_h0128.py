"""H-0128 negative controls: undo one part of the change, run the test file, and
check that it fails. Every file is restored after each mutation. Run from the
repository root, through instruments/heavy.sh."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MT = Path("lizyml/core/_model_tuning.py")
PLAN = Path("lizyml/core/_tuning_plan.py")
TUNER = Path("lizyml/tuning/tuner.py")
TF = Path("lizyml/tuning/trial_failure.py")
REST = Path("lizyml/core/_tuning_restore.py")
TESTS = "tests/test_tuning/test_h0128_failed_rounds.py"

MUTATIONS = [
    (
        "criterion 1/3: the round closes on a study-wide COMPLETE (H-0122 rule)",
        MT,
        "t.state.name == \"COMPLETE\" and t.number >= plan.start_trial",
        "t.state.name == \"COMPLETE\"",
    ),
    (
        "criterion 3: interrupted trials not counted",
        MT,
        "return len(plan.running_to_fail) if plan.kind == \"open\" else 0",
        "return 0",
    ),
    (
        "criterion 4: the public Tuner.tune bounds by prior_trials",
        TUNER,
        "boundary = len(study.trials)",
        "boundary = prior_trials",
    ),
    (
        "criterion 5: the provider's trial errors are not passed",
        MT,
        "        return provider.trial_errors()\n    return ()",
        "        return ()\n    return ()",
    ),
    (
        "criterion 5b: the last error is kept instead of the first",
        TUNER,
        "                if self.first_error is None:\n                    self.first_error = text",
        "                self.first_error = text",
    ),
    (
        "criterion 5b: str(exc) used unguarded",
        TF,
        "    try:\n        message = str(exc)",
        "    message = str(exc)\n    try:\n        message = str(exc)",
    ),
    (
        "criterion 5c: trial_errors() called on every provider",
        MT,
        "    if isinstance(provider, TrialErrorProvider):\n        return provider.trial_errors()",
        "    if True:\n        return provider.trial_errors()",
    ),
    (
        "criterion 6: no replacement (U2 refuses as before)",
        PLAN,
        "    if same_declared_space(declaration.declared, record.declared, lineage.routes):\n        return False",
        "    if True:\n        return False",
    ),
    (
        "criterion 7: RUNNING trials failed while planning (a write before a refusal)",
        PLAN,
        "            return dataclasses.replace(",
        "            from lizyml.tuning import tuner as _t\n\n            _t.fail_running_trials(study, running)\n            return dataclasses.replace(",
    ),
    (
        "criterion 8: the old restore refusal text",
        REST,
        "f\"its last round ({last}) closed failed: no trial of that round \"",
        "\"its last round closed failed (no trial completed). \"",
    ),
]

not_caught = []
for label, path, old, new in MUTATIONS:
    original = path.read_text(encoding="utf-8")
    if original.count(old) != 1:
        print(f"[setup error] {label}: pattern found {original.count(old)} times in {path}")
        not_caught.append(label)
        continue
    path.write_text(original.replace(old, new, 1), encoding="utf-8")
    try:
        run = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-x", TESTS],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        path.write_text(original, encoding="utf-8")
    tail = run.stdout.strip().splitlines()[-1] if run.stdout.strip() else run.stderr[-200:]
    verdict = "FAILS as required" if run.returncode != 0 else "STILL PASSES"
    print(f"{label}: {verdict} ({tail})")
    if run.returncode == 0:
        not_caught.append(label)

print("all mutations caught" if not not_caught else f"not caught: {not_caught}")
sys.exit(1 if not_caught else 0)
