"""PR 8c measurement: run each candidate `derived_from` snippet the way the tool will.

Each snippet is passed to `python -c` with cwd = the tree under test, which is how
the shipped runner evaluates it (a script file would import the main checkout;
see pr8c_measurements.txt item 3). Prints issue, value, and the snippet's first line.

    .venv/bin/python .../pr8c_derivations.py <tree>
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

SNIPPETS = {
    "258": "from lizyml.metrics.registry import _TASK_METRICS\n"
           "print(sum(len(v) for v in _TASK_METRICS.values()))",
    "259": "from lizyml.features.pipeline_base import BaseFeaturePipeline\n"
           "print(len(BaseFeaturePipeline.__abstractmethods__))",
    "260": "import typing\n"
           "from lizyml.features.encoders.categorical_encoder import UnseenPolicy\n"
           "print(len(typing.get_args(UnseenPolicy)))",
    "261": "from lizyml.core.types.task import TASK_TYPES\nprint(len(TASK_TYPES))",
    "262": "import typing\nfrom lizyml.core.types.task import TASK_TYPES\n"
           "from lizyml.core.types.search_dim import DimCategory\n"
           "print(3 * len(typing.get_args(DimCategory)) * len(TASK_TYPES))",
    "263": "from lizyml.core.exceptions import ErrorCode\nprint(len(ErrorCode.__members__))",
    "269": "import inspect\nfrom lizyml.training.cv_trainer import CVTrainer\n"
           "from lizyml.training.refit_trainer import RefitTrainer\n"
           "a = set(inspect.signature(CVTrainer.fit).parameters) - {'self'}\n"
           "b = set(inspect.signature(RefitTrainer.fit).parameters) - {'self'}\n"
           "print(len(a | b))",
    "271": "import pathlib, re\nt = pathlib.Path('HISTORY.md').read_text(encoding='utf-8')\n"
           "ids = set(re.findall(r'^## (H-\\d{4})\\b', t, re.M))\n"
           "ids |= set(re.findall(r'^\\s*-\\s*ID:\\s*`?(H-\\d{4})', t, re.M))\nprint(len(ids))",
    "277": "from lizyml.calibration.registry import CalibratorRegistry\n"
           "print(len(CalibratorRegistry.keys()))",
    "306": "from lizyml.estimators.lgbm.defaults import TASK_COMPATIBLE_OBJECTIVES\n"
           "from lizyml.estimators.lgbm.metric_bridge import _FEVAL_METRICS\n"
           "print(sum(len(TASK_COMPATIBLE_OBJECTIVES[t]) * len(_FEVAL_METRICS[t])"
           " for t in TASK_COMPATIBLE_OBJECTIVES))",
}


def main(tree: str) -> int:
    py = str(pathlib.Path(".venv/bin/python").absolute())
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for issue, code in SNIPPETS.items():
        p = subprocess.run([py, "-c", code], cwd=tree, capture_output=True, text=True,
                           env=env)
        value = p.stdout.strip() if p.returncode == 0 else f"ERROR {p.stderr.strip()[-160:]}"
        print(f"#{issue}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
