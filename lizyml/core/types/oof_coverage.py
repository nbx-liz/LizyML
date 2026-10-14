"""Which rows the outer validation folds cover (H-0057, moved to Layer 0 by H-0130).

The mask is derived from split indices alone, so both ``training/`` and
``evaluation/`` read it from Foundation instead of importing each other
(BLUEPRINT §2.1). ``training/oof_assembly.py`` re-exports it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["compute_oof_valid_mask"]


def compute_oof_valid_mask(
    splits_outer: list[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]],
    n_samples: int,
) -> npt.NDArray[np.bool_]:
    """Return boolean mask: True for rows covered by at least one validation fold.

    Derives coverage deterministically from split indices — not from NaN
    detection — so that bugs (missing predictions in covered rows) are
    distinguishable from structurally uncovered rows (e.g. first period
    in TimeSeriesCV).

    Args:
        splits_outer: Per-fold ``(train_idx, valid_idx)`` tuples.
        n_samples: Total number of samples in the dataset.

    Returns:
        Boolean array of shape ``(n_samples,)``.
    """
    mask = np.zeros(n_samples, dtype=np.bool_)
    for _, valid_idx in splits_outer:
        mask[valid_idx] = True
    return mask
