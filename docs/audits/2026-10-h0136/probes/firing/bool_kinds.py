"""H-0136 (review round 2): Python ``True`` versus ``np.bool_(True)`` beside ``np.uint64(2**63 + 5)``.

``criterion4_kinds.py`` named types by ``type(v).__name__``, which is
``bool`` for both, so its ``bool + uint64`` rows could not say which one
breaks. This probe runs the prototype rule on each pair, plus the
uint64/timedelta64 pairs criterion 4 names.

Usage: python bool_kinds.py   (this directory and the repository on the path)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from h0136_rule import state_breaks

from lizyml.features.encoders.categorical_encoder import CategoricalEncoder

warnings.simplefilter("ignore")
CASES = {
    "np.uint64(2**63+5) + True": [np.uint64(2**63 + 5), True],
    "np.uint64(2**63+5) + np.bool_(True)": [np.uint64(2**63 + 5), np.bool_(True)],
    "np.uint64(4) + np.timedelta64(4, 'M')": [np.uint64(4), np.timedelta64(4, "M")],
}
print(f"pandas={pd.__version__} numpy={np.__version__}")
for name, values in CASES.items():
    df = pd.DataFrame({"c": pd.Series([*values, *values, "x"], dtype=object)})
    state = {"encoder": CategoricalEncoder().fit(df, ["c"]).get_state()}
    print(f"  {name}: {state_breaks(state)[0]}")
