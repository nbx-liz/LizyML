"""In-suite positive control for firing_rate_344.py: run with the plugin, e.g.

    FIRING_RATE_344_OUT=out.tsv PYTHONPATH=docs/audits/2026-10-tuning-resume/probes \\
        .venv/bin/python -m pytest -p firing_rate_344 -p no:cacheprovider \\
        docs/audits/2026-10-tuning-resume/probes/firing_rate_344_posctl.py

The plugin must record reasons on both production paths (config space, space=).
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from lizyml import Model
from tests.test_tuning._resume_helpers import SPACE, fast_config, frame


def test_config_path():
    cfg = fast_config("regression", n_trials=1)
    cfg["tuning"]["optuna"]["space"]["num_leaves"]["step"] = 2
    cfg["tuning"]["optuna"]["space"]["learning_rate"]["log"] = "false"
    Model(cfg, data=frame()).tune()


def test_space_arg(tmp_path):
    cfg = fast_config("regression", n_trials=1)
    url = f"sqlite:///{tmp_path / 's.db'}"
    Model(copy.deepcopy(cfg), data=frame()).tune(storage=url, study_name="s")
    space = copy.deepcopy(SPACE)
    space["num_leaves"]["low"] = "1e0x"
    try:
        Model(copy.deepcopy(cfg), data=frame()).tune(resume=True, storage=url, study_name="s", expand_boundary=False, space=space)
    except Exception:
        pass
