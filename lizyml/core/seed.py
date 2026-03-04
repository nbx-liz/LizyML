"""Seed management utilities for reproducibility."""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set the global random seed for numpy and random.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)


def derive_seed(base_seed: int, fold_index: int) -> int:
    """Derive a deterministic seed for a given fold.

    Guarantees that seeds differ across folds while remaining
    fully reproducible given the same base_seed.

    Args:
        base_seed: The root seed (e.g. from config).
        fold_index: Zero-based fold index.

    Returns:
        A deterministic integer seed for this fold.
    """
    # Use a simple but collision-resistant mixing function.
    # The constant is a large prime to spread the values well.
    return (base_seed * 1_000_003 + fold_index) % (2**31)
