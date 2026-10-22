"""Does an integer column name in categorical_feature select the wrong column?"""

import warnings

import numpy as np
import pandas as pd

from lizyml import Model
from tests._helpers import make_config

warnings.simplefilter("ignore")
rng = np.random.default_rng(0)
n = 180
for label in ["c", 7, 0, 1]:
    cat = rng.choice(["a", "b", "c"], size=n)
    df = pd.DataFrame({label: pd.Series(cat, dtype=object), "x": rng.normal(size=n)})
    df["target"] = (pd.Series(cat) == "a").astype(float) + rng.normal(scale=0.1, size=n)
    model = Model(make_config("regression", n_estimators=3, n_splits=2))
    model.fit(data=df)
    booster = model._refit_result.model.get_native_model()
    print(
        f"label {label!r}: feature_name={booster.feature_name()} "
        f"categorical_column={booster.params.get('categorical_column')} "
        f"pandas_categorical={booster.pandas_categorical}"
    )
