"""Every defaulted constructor knob of a public class, and where its value comes from.

H-0108 (#268). The population is computed at test time by an AST sweep
(``knob_census`` below); this table must classify exactly that population, so a
new knob fails until someone decides where its value comes from.

Classification rule -- the first kind that applies, read on every production
path that constructs the class:

``config``
    A Config key's value is passed to the argument (possibly only on some
    paths, or with a stated fallback). The detail names each path's source.
    Each row has at least one executed cell in ``test_knob_reachability.py``,
    and every path with a different source has its own cell.
``api``
    No Config key sets it; a public call argument does (``Model(...)``,
    ``Model.tune(...)``). Stated in ``BLUEPRINT.md`` §5.5.
``derived``
    The library computes it from the data, or from settings through a rule
    that is not passing one Config value on. Stated in §5.5 with the rule.
``policy``
    The library fixes the value: it passes a constant or leaves the default,
    and no Config key or public argument changes it. Stated in §5.5 with the
    reason.
``internal``
    Not a behaviour setting (error payload, wiring between components). Stated
    in §5.5.
"""

from __future__ import annotations

import ast
from pathlib import Path

import lizyml

_PACKAGE = Path(lizyml.__file__).resolve().parent

KINDS = ("config", "api", "derived", "policy", "internal")


def knob_census() -> set[str]:
    """``Class.param`` for every defaulted or keyword-only ``__init__`` parameter
    of every class under ``lizyml/`` whose name does not start with ``_``."""
    knobs: set[str] = set()
    for path in _PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
                continue
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == "__init__":
                    a = sub.args
                    for arg in a.args[len(a.args) - len(a.defaults) :]:
                        knobs.add(f"{node.name}.{arg.arg}")
                    for arg in a.kwonlyargs:
                        knobs.add(f"{node.name}.{arg.arg}")
    return knobs


_SEED_FALLBACK = "split.random_state; training.seed when split.random_state is None"
_AUTO_SEED = (
    "explicit inner_valid: inner_valid.random_state;"
    " resolved automatically: training.seed"
)
_RATIO = (
    "explicit inner_valid: inner_valid.ratio; resolved automatically:"
    " early_stopping.validation_ratio (0.1 when absent)"
)

#: knob -> (kind, detail). The detail is what BLUEPRINT §5.5 states for the
#: kinds other than ``config``.
REGISTRY: dict[str, tuple[str, str]] = {
    # calibration
    "PlattCalibrator.params": ("config", "calibration.params"),
    "IsotonicCalibrator.params": ("config", "calibration.params (plus the seed)"),
    "BetaCalibrator.params": ("config", "calibration.params"),
    # errors
    "LizyMLError.debug_message": ("internal", "error payload, set by each raise site"),
    "LizyMLError.cause": ("internal", "error payload, set by each raise site"),
    "LizyMLError.context": ("internal", "error payload, set by each raise site"),
    # facade
    "Model.data": ("api", "Model(config, data=...)"),
    "Model.output_dir": (
        "config",
        "output_dir; the constructor argument, when given, takes priority",
    ),
    # estimator adapter
    "LGBMAdapter.task": ("config", "task"),
    "LGBMAdapter.params": ("config", "model.params (merged with resolved params)"),
    "LGBMAdapter.num_class": (
        "derived",
        "number of target classes; None unless multiclass",
    ),
    "LGBMAdapter.early_stopping_rounds": ("config", "training.early_stopping.rounds"),
    "LGBMAdapter.verbose_eval": ("policy", "-1: per-iteration evaluation log off"),
    "LGBMAdapter.random_state": ("config", "training.seed"),
    # evaluation
    "Evaluator.task": ("config", "task"),
    # features
    "CategoricalEncoder.unseen_policy": ("config", "features.unseen_policy"),
    "NativeFeaturePipeline.unseen_policy": (
        "config",
        "features.unseen_policy (restored from saved state where state is loaded)",
    ),
    # metrics (dict form of a metric entry, H-0065)
    "ECE.n_bins": ("config", "evaluation.metrics / model.params metric (feval)"),
    "PrecisionAtK.k": ("config", "evaluation.metrics / model.params metric (feval)"),
    "HuberLoss.delta": (
        "config",
        "evaluation.metrics (a model.params metric entry is native; #313)",
    ),
    # splitters
    "KFoldSplitter.n_splits": ("config", "split.n_splits"),
    "KFoldSplitter.shuffle": ("config", "split.shuffle"),
    "KFoldSplitter.random_state": ("config", _SEED_FALLBACK),
    "StratifiedKFoldSplitter.n_splits": ("config", "split.n_splits"),
    "StratifiedKFoldSplitter.shuffle": ("policy", "True: always shuffles"),
    "StratifiedKFoldSplitter.random_state": ("config", _SEED_FALLBACK),
    "GroupKFoldSplitter.n_splits": ("config", "split.n_splits"),
    "StratifiedGroupKFoldSplitter.n_splits": ("config", "split.n_splits"),
    "StratifiedGroupKFoldSplitter.shuffle": ("config", "split.shuffle"),
    "StratifiedGroupKFoldSplitter.random_state": ("config", _SEED_FALLBACK),
    "TimeSeriesSplitter.n_splits": ("config", "split.n_splits"),
    "TimeSeriesSplitter.gap": ("config", "split.gap"),
    "TimeSeriesSplitter.max_train_size": ("config", "split.train_size_max"),
    "TimeSeriesSplitter.max_test_size": ("config", "split.test_size_max"),
    "PurgedTimeSeriesSplitter.n_splits": ("config", "split.n_splits"),
    "PurgedTimeSeriesSplitter.purge_gap": ("config", "split.purge_gap"),
    "PurgedTimeSeriesSplitter.embargo": ("config", "split.embargo"),
    "PurgedTimeSeriesSplitter.max_train_size": ("config", "split.train_size_max"),
    "PurgedTimeSeriesSplitter.max_test_size": ("config", "split.test_size_max"),
    "GroupTimeSeriesSplitter.n_splits": ("config", "split.n_splits"),
    "GroupTimeSeriesSplitter.gap": ("config", "split.gap"),
    "GroupTimeSeriesSplitter.max_train_size": ("config", "split.train_size_max"),
    "GroupTimeSeriesSplitter.max_test_size": ("config", "split.test_size_max"),
    "BlockedGroupKFoldSplitter.mode": ("config", "split.blocks.mode"),
    "BlockedGroupKFoldSplitter.train_window": ("config", "split.blocks.train_window"),
    "BlockedGroupKFoldSplitter.n_splits": ("config", "split.groups.n_splits"),
    "BlockedGroupKFoldSplitter.stratify": ("config", "split.groups.stratify"),
    "BlockedGroupKFoldSplitter.shuffle": ("config", "split.groups.shuffle"),
    "BlockedGroupKFoldSplitter.random_state": ("config", "training.seed"),
    "BlockedGroupKFoldSplitter.min_train_rows": ("config", "split.min_train_rows"),
    "BlockedGroupKFoldSplitter.min_valid_rows": ("config", "split.min_valid_rows"),
    # trainers
    "CVTrainer.task": ("config", "task"),
    "CVTrainer.n_classes": (
        "derived",
        "number of target classes; None unless multiclass",
    ),
    "CVTrainer.ratio_param_resolver": (
        "internal",
        "wiring: resolves ratio-form smart params per fold",
    ),
    "CVTrainer.collect_raw_scores": (
        "derived",
        "fit: whether calibration is configured; tune trials: False",
    ),
    "RefitTrainer.task": ("config", "task"),
    "RefitTrainer.ratio_param_resolver": (
        "internal",
        "wiring: resolves ratio-form smart params",
    ),
    # inner validation
    "HoldoutInnerValid.ratio": ("config", _RATIO),
    "HoldoutInnerValid.random_state": ("config", _AUTO_SEED),
    "HoldoutInnerValid.stratify": (
        "config",
        "explicit inner_valid: inner_valid.stratify; resolved automatically:"
        " True for stratified_kfold, False otherwise",
    ),
    "GroupHoldoutInnerValid.ratio": ("config", _RATIO),
    "GroupHoldoutInnerValid.random_state": ("config", _AUTO_SEED),
    "TimeHoldoutInnerValid.ratio": ("config", _RATIO),
    "TimeHoldoutInnerValid.gap": (
        "config",
        "resolved automatically: split.gap (time_series) or split.purge_gap +"
        " split.embargo (purged_time_series); 0 for an explicit time_holdout"
        " (H-0101) and on BlockedGroupInnerValid's regression fallback",
    ),
    "StratifiedTimeHoldoutInnerValid.ratio": (
        "config",
        "BlockedGroupInnerValid's ratio (early_stopping.validation_ratio),"
        " on its classification fallback",
    ),
    "BlockedGroupInnerValid.ratio": ("config", _RATIO),
    "BlockedGroupInnerValid.task": ("config", "task"),
    # tuning
    "Tuner.n_trials": ("config", "tuning.optuna.params.n_trials"),
    "Tuner.direction": ("config", "tuning.optuna.params.direction"),
    "Tuner.timeout": ("config", "tuning.optuna.params.timeout"),
    "Tuner.seed": ("config", "training.seed"),
    "Tuner.progress_callback": ("api", "Model.tune(progress_callback=...)"),
    "Tuner.storage": ("api", "Model.tune(storage=...)"),
    "Tuner.study_name": ("api", "Model.tune(study_name=...)"),
}
