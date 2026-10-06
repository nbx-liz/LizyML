"""H-0111 reference count: default-space dims that a re-tune can leave pinned.

For each task, attach the LightGBM provider's parameter bounds to its default
search space and count the numeric dims whose current ``low`` already equals
``min_allowed`` or whose ``high`` already equals ``max_allowed``. Such a dim,
with its best value at that edge, makes ``detect_boundary`` compute an
expansion that leaves the range unchanged -- the case H-0111 stops reporting
as ``expanded``. Run from the repo root:

    uv run python docs/audits/2026-09-defect-discovery/instruments/h0111_pinned_dims.py
"""

from __future__ import annotations

from lizyml.core.types.search_dim import CategoricalDim
from lizyml.estimators.lgbm.provider import LGBMProvider
from lizyml.tuning.search_space import attach_bounds

TASKS = ("regression", "binary", "multiclass")


def main() -> None:
    provider = LGBMProvider()
    for task in TASKS:
        dims = attach_bounds(provider.default_space(task), provider.parameter_bounds(task))
        numeric = [d for d in dims if not isinstance(d, CategoricalDim)]
        pinned = []
        for d in numeric:
            if d.min_allowed is not None and float(d.low) == float(d.min_allowed):
                pinned.append(f"{d.name}:low")
            if d.max_allowed is not None and float(d.high) == float(d.max_allowed):
                pinned.append(f"{d.name}:high")
        print(f"{task}: {len(pinned)} pinned edges over {len(numeric)} numeric dims {pinned}")


if __name__ == "__main__":
    main()
