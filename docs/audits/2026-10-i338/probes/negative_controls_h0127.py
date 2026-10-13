"""H-0127 negative controls: undo one part of the change, run the tests that
claim to guard it, and check that they fail. Every file is restored after each
mutation. Run from the repository root with the project's venv python."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ENC = Path("lizyml/features/encoders/categorical_encoder.py")
EXP = Path("lizyml/persistence/exporter.py")
TPL = Path("lizyml/codegen/templates.py")
MOD = Path("lizyml/core/model.py")
CAT = Path("lizyml/estimators/lgbm/category_json.py")

MUTATIONS = [
    (
        "NaN counted as changed (criterion 2b)",
        ENC,
        "changed = ~np.isnan(values) & (as64.astype(dtype) != values)",
        "changed = as64.astype(dtype) != values",
        "tests/test_features/test_categorical_float_arrivals.py",
    ),
    (
        "lossy longdouble matched after rounding (criterion 2)",
        ENC,
        "hit = series.isin(list(unseen)) | changed",
        "hit = series.isin(list(unseen))",
        "tests/test_features/test_categorical_float_arrivals.py",
    ),
    (
        "train.py copy left without the normalisation (criterion 3)",
        TPL,
        # The first occurrence is train.py's copy; predict.py's stays fixed.
        "    arrived = series\n    series, changed = comparable_values(arrived)\n",
        "    arrived = series\n    changed = pd.Series(False, index=series.index)\n",
        "tests/test_codegen/test_float_arrival_codegen.py",
    ),
    (
        "export check removed (criterion 5)",
        EXP,
        "    check_category_values(fit_result, refit_result)\n    out = Path(path)",
        "    out = Path(path)",
        "tests/test_persistence/test_unwritable_categories.py",
    ),
    (
        "export check reads the refit state only (criterion 5b)",
        EXP,
        "    if per_fold is None:",
        "    return named\n    if per_fold is None:",
        "tests/test_persistence/test_unwritable_categories.py",
    ),
    (
        "fit warning removed (criterion 6)",
        MOD,
        "        warn_unsaveable_categories(pipeline_factory().fit(X, y).get_state())\n",
        "",
        "tests/test_persistence/test_unwritable_categories.py",
    ),
    (
        "plain repr in the formatter (criterion 4b)",
        CAT,
        "    try:\n        text = repr(value)\n    except Exception:  # noqa: BLE001 - a hostile __repr__ must not stop fit\n        return f\"<unprintable {type_name(value)}>\"\n",
        "    text = repr(value)\n",
        "tests/test_persistence/test_unwritable_categories.py",
    ),
    (
        "numpy scalars with a non-plain .item() accepted (criterion 4)",
        CAT,
        "        if isinstance(plain, np.generic):\n            raise TypeError(type_name(obj))\n",
        "",
        "tests/test_persistence/test_unwritable_categories.py",
    ),
]

failed_to_fail = []
for label, path, old, new, tests in MUTATIONS:
    original = path.read_text(encoding="utf-8")
    if original.count(old) < 1:
        print(f"[setup error] {label}: pattern not found in {path}")
        failed_to_fail.append(label)
        continue
    path.write_text(original.replace(old, new, 1), encoding="utf-8")
    try:
        run = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", tests],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        path.write_text(original, encoding="utf-8")
    summary = run.stdout.strip().splitlines()[-1] if run.stdout.strip() else run.stderr[-200:]
    verdict = "FAILS as required" if run.returncode != 0 else "STILL PASSES"
    print(f"{label}: {verdict} ({summary})")
    if run.returncode == 0:
        failed_to_fail.append(label)

print("all mutations caught" if not failed_to_fail else f"not caught: {failed_to_fail}")
sys.exit(1 if failed_to_fail else 0)
