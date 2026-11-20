"""Does optuna.load_study write anything to an RDB study? Fingerprint every table before/after."""

import hashlib
import sqlite3
import sys

import optuna

optuna.logging.set_verbosity(optuna.logging.ERROR)
db = sys.argv[1]
url = f"sqlite:///{db}"


def fingerprint() -> str:
    con = sqlite3.connect(db)
    h = hashlib.sha256()
    tables = [r[0] for r in con.execute("select name from sqlite_master where type='table' order by name")]
    for t in tables:
        h.update(t.encode())
        for row in con.execute(f"select * from {t} order by 1"):
            h.update(repr(row).encode())
    con.close()
    return h.hexdigest()[:16]


study = optuna.create_study(study_name="s", storage=url, sampler=optuna.samplers.TPESampler(seed=1))
study.set_user_attr("k", "v")
study.optimize(lambda t: t.suggest_float("x", 0, 1) ** 2, n_trials=5)
del study
before = fingerprint()
loaded = optuna.load_study(study_name="s", storage=url, sampler=optuna.samplers.TPESampler(seed=1))
_ = loaded.trials, loaded.user_attrs, loaded.best_trial
after = fingerprint()
print(optuna.__version__, before, after, "UNCHANGED" if before == after else "CHANGED")
