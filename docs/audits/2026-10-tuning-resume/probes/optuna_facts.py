"""Verify the Optuna behaviours H-0122 PR 1 depends on (DC7: run the real thing)."""

import os
import pickle
import subprocess
import sys
import tempfile

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
print("optuna", optuna.__version__)
tmp = tempfile.mkdtemp()
url = f"sqlite:///{tmp}/s.db"

# 1. A RUNNING trial left by a dead process.
child = f"""
import optuna, os
s = optuna.create_study(storage={url!r}, study_name="x")
s.set_user_attr("lizyml:study", {{"schema": 1}})
s.optimize(lambda t: t.suggest_float("a", 0, 1), n_trials=2)
t = s.ask()
t.suggest_float("a", 0, 1)
os._exit(0)
"""
subprocess.run([sys.executable, "-c", child], check=True)
s = optuna.load_study(storage=url, study_name="x")
print("1 states before:", [t.state.name for t in s.trials])
print("1 user_attrs:", s.user_attrs)
running = [t for t in s.trials if t.state == optuna.trial.TrialState.RUNNING]
s2 = optuna.load_study(storage=url, study_name="x")
print("1 still RUNNING after second load (load changes nothing):",
      [t.state.name for t in s2.trials])
s.tell(running[0].number, state=optuna.trial.TrialState.FAIL)
s3 = optuna.load_study(storage=url, study_name="x")
print("1 after tell FAIL:", [(t.number, t.state.name, t.datetime_start is not None,
      t.datetime_complete is not None) for t in s3.trials])

# 2. PRUNED trials carry start/complete times.
def obj(t):
    t.suggest_float("a", 0, 1)
    if t.number % 2:
        raise optuna.TrialPruned()
    return 1.0

p = optuna.create_study()
p.optimize(obj, n_trials=4)
print("2 pruned times:", [(t.state.name, t.datetime_start is not None,
      t.datetime_complete is not None) for t in p.trials])

# 3. in-memory user attrs survive pickle.
m = optuna.create_study()
m.set_user_attr("lizyml:study", {"schema": 1, "objective": {"d": "abc"}})
m.optimize(lambda t: t.suggest_float("a", 0, 1), n_trials=1)
m2 = pickle.loads(pickle.dumps(m))
print("3 pickled attrs:", m2.user_attrs, len(m2.trials))

# 4. study names / existence check.
print("4 names:", optuna.get_all_study_names(url))

# 5. timeout semantic: does optimize(timeout=0) start a trial?
z = optuna.create_study()
z.optimize(lambda t: t.suggest_float("a", 0, 1), n_trials=5, timeout=0.0)
print("5 trials with timeout=0:", len(z.trials))
