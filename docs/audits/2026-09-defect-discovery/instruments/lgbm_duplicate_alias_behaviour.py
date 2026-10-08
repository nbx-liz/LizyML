"""Does LightGBM warn on duplicate aliases?  The earlier probes passed
`verbose: -1`, which suppresses the C++ log, so "silent" was not measured.
This one leaves verbosity at the default and captures the C++ stderr/stdout
at the file-descriptor level.
"""

import os
import sys
import tempfile

import numpy as np
import lightgbm as lgb

X = np.random.RandomState(0).rand(80, 4)
y = (X[:, 0] > 0.5).astype(int)


def train_capturing(params):
    ds = lgb.Dataset(X, label=y)
    with tempfile.TemporaryFile(mode="w+") as tmp:
        sys.stdout.flush()
        sys.stderr.flush()
        saved_out, saved_err = os.dup(1), os.dup(2)
        os.dup2(tmp.fileno(), 1)
        os.dup2(tmp.fileno(), 2)
        try:
            booster = lgb.train({"objective": "binary", "num_leaves": 3, **params}, ds, num_boost_round=1)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)
        tmp.seek(0)
        log = tmp.read()
    used = "?"
    for line in booster.model_to_string().splitlines():
        if line.startswith("[learning_rate:"):
            used = line.strip()
    return used, log


CASES = [
    ("canonical + alias, different", {"learning_rate": 0.5, "eta": 0.9}),
    ("canonical + alias, equal", {"learning_rate": 0.5, "eta": 0.5}),
    ("two aliases, different", {"eta": 0.2, "shrinkage_rate": 0.3}),
    ("two aliases, reversed order", {"shrinkage_rate": 0.3, "eta": 0.2}),
    ("single spelling (control)", {"learning_rate": 0.5}),
]

for label, params in CASES:
    used, log = train_capturing(params)
    interesting = [
        ln for ln in log.splitlines()
        if "ignor" in ln.lower() or "warn" in ln.lower() or "duplicat" in ln.lower()
    ]
    print(f"--- {label}: {used}")
    for ln in interesting:
        print(f"      {ln.strip()}")
    if not interesting:
        print(f"      (no ignore/warning line; log had {len(log.splitlines())} lines)")
