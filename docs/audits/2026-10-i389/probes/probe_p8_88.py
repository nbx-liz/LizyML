"""Why does p8-88 order inf before datetime64(2s) on pandas 3.1.0?"""

from __future__ import annotations

import pandas as pd

from tests.test_features import _h0138_oracle as h
from tests.test_features._h0138_answers import temporal_answer

values = h.probe_values("p8-88")
print("values", [(type(v).__name__, repr(v), str(v)) for v in values])
train = pd.Series([*values, *values, "x"], dtype=object)
uniq = train.dropna().unique().tolist()
print("unique", [(type(v).__name__, repr(v), str(v)) for v in uniq])
print("sorted by str", [repr(v) for v in sorted(uniq, key=str)])
print("answer", temporal_answer(train, values))
