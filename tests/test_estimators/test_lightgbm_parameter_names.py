"""Every name LizyML hands to LightGBM must be a name LightGBM defines.

A parameter LightGBM does not recognise is **discarded silently**. Nothing
raises, the booster trains, and the run looks exactly like one where the
parameter was honoured -- LightGBM's own warning about unknown keys is
suppressed by ``verbose=-1``, which is the shipped default. So a misspelled or
invented name is inert in the one way a test asserting "the fit succeeded"
cannot see.

That is not hypothetical here: ``feature_weights`` was emitted for the whole
life of the smart parameter, and LightGBM 4.6.0 has no such name or alias
(``feature_contri`` is the one it defines). ``BLUEPRINT.md:1425`` declares
"``feature_weights`` changes the importance ordering" as an invariant to be
verified, and it was false of the shipped code (H-0093).

**The authority is LightGBM, not a list in this file.** Train parameters are
checked against ``LGBM_DumpParamAliases``, the table the library dumps from its
own registry, and ``lgb.Dataset`` keywords against that constructor's real
signature. A hand-written copy of either would go stale exactly when LightGBM
changes -- the drift this check exists to catch.
"""

from __future__ import annotations

import ast
import ctypes
import inspect
import json
import pathlib
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import lightgbm as lgb
import pytest

from lizyml.config.loader import load_config
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.model import Model
from lizyml.core.types.tuning_result import TuningResult
from lizyml.estimators.lgbm.provider import LGBMProvider
from tests._ast_scan import LightGBMBindings, lightgbm_bindings
from tests._helpers import (
    make_binary_df,
    make_config,
    make_multiclass_df,
    make_regression_df,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
TASKS = ("regression", "binary", "multiclass")


# --------------------------------------------------------------------------
# The authorities
# --------------------------------------------------------------------------
def _lightgbm_param_names() -> tuple[frozenset[str], frozenset[str]]:
    """Return ``(canonical, canonical | aliases)`` from LightGBM's own table."""
    size = 1 << 20
    buf = ctypes.create_string_buffer(size)
    out_len = ctypes.c_int64(0)
    lgb.basic._LIB.LGBM_DumpParamAliases(
        ctypes.c_int64(size), ctypes.byref(out_len), ctypes.byref(buf)
    )
    table: dict[str, list[str]] = json.loads(buf.value.decode("utf-8"))
    every = set(table)
    for aliases in table.values():
        every.update(aliases)
    return frozenset(table), frozenset(every)


CANONICAL, LGBM_NAMES = _lightgbm_param_names()

#: Keywords ``lgb.Dataset`` actually accepts, from its signature.
DATASET_KWARGS = frozenset(inspect.signature(lgb.Dataset.__init__).parameters) - {
    "self"
}


def test_the_authorities_are_non_trivial() -> None:
    """A guard against checking against an empty set (DC1).

    If the ctypes dump or the signature introspection ever returns nothing,
    every assertion below would pass vacuously.
    """
    assert len(CANONICAL) > 100, f"only {len(CANONICAL)} canonical names dumped"
    assert len(LGBM_NAMES) > len(CANONICAL), "aliases missing from the dump"
    assert "num_leaves" in CANONICAL, "the dump does not look like LightGBM's table"
    assert {"label", "weight", "categorical_feature"} <= DATASET_KWARGS


def test_the_shipped_derivation_agrees_with_this_one() -> None:
    """The package derives the same set this file derives.

    The derivation above is deliberately independent -- a check that asked the
    code under test what the right answer is could not detect a broken
    derivation. This one assertion ties the two together, so a change to
    ``param_names.py`` that narrows or widens the set fails here rather than
    silently changing what the gate accepts.
    """
    assert LGBMProvider().accepted_model_param_names() == LGBM_NAMES


# --------------------------------------------------------------------------
# Smart parameters: derived from the provider, never listed
# --------------------------------------------------------------------------
class _EmptyModelCfg:
    """Stand-in whose attributes are the smart fields, all unset."""

    auto_num_leaves = None
    num_leaves_ratio = None
    min_data_in_leaf_ratio = None
    min_data_in_bin_ratio = None
    feature_weights = None
    balanced = None


DECLARED_SMART = frozenset(LGBMProvider().extract_smart_params(_EmptyModelCfg()))

#: A value that *activates* each smart parameter. ``balanced`` is classification
#: only -- the provider raises ``UNSUPPORTED_TASK`` for regression -- so it
#: carries the tasks it applies to.
SMART_VALUES: dict[str, tuple[Any, tuple[str, ...]]] = {
    "auto_num_leaves": (True, TASKS),
    "num_leaves_ratio": (0.5, TASKS),
    "min_data_in_leaf_ratio": (0.02, TASKS),
    "min_data_in_bin_ratio": (0.01, TASKS),
    "feature_weights": ({"feat_a": 2.0}, TASKS),
    "balanced": (True, ("binary", "multiclass")),
}


def test_every_smart_parameter_has_a_case() -> None:
    """The cases must cover what the provider declares, not a sample of it.

    Listing values by hand turns the population into a sample that claims to be
    a population. This fails when the provider grows a smart parameter that no
    case here activates.
    """
    missing = sorted(DECLARED_SMART - set(SMART_VALUES))
    extra = sorted(set(SMART_VALUES) - DECLARED_SMART)
    assert not missing, (
        f"the provider declares smart parameters with no case here: {missing}. "
        "Add a value that activates each one, so its emitted keys are checked."
    )
    assert not extra, (
        f"cases exist for smart parameters the provider no longer declares: "
        f"{extra}. Remove them, or the check is measuring a name nothing emits."
    )


# --------------------------------------------------------------------------
# Recording what actually reaches LightGBM
# --------------------------------------------------------------------------
@contextmanager
def _record_lightgbm_calls() -> Iterator[dict[str, list[Any]]]:
    """Capture every ``lgb.train`` params dict and ``lgb.Dataset`` keyword.

    Patches the module attributes the adapter resolves at call time
    (``lizyml/estimators/lgbm/adapter.py`` holds ``import lightgbm as lgb`` and
    calls ``lgb.train`` / ``lgb.Dataset``), so the real functions still run.
    """
    seen: dict[str, list[Any]] = {"train_params": [], "dataset_kwargs": []}
    real_train = lgb.train
    real_dataset = lgb.Dataset

    def spy_train(params: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        seen["train_params"].append(dict(params))
        return real_train(params, *args, **kwargs)

    def spy_dataset(*args: Any, **kwargs: Any) -> Any:
        seen["dataset_kwargs"].append(sorted(kwargs))
        return real_dataset(*args, **kwargs)

    lgb.train = spy_train  # type: ignore[assignment]
    lgb.Dataset = spy_dataset  # type: ignore[assignment,misc]
    try:
        yield seen
    finally:
        lgb.train = real_train  # type: ignore[assignment]
        lgb.Dataset = real_dataset  # type: ignore[assignment,misc]


def _df_for(task: str) -> Any:
    if task == "regression":
        return make_regression_df(n=120)
    if task == "binary":
        return make_binary_df(n=120)
    return make_multiclass_df(n=150)


def _fit_with_every_smart_parameter(task: str) -> dict[str, list[Any]]:
    """Fit once with every smart parameter that applies to *task* activated."""
    overrides = {
        name: value for name, (value, tasks) in SMART_VALUES.items() if task in tasks
    }
    cfg = make_config(task, n_estimators=5, n_splits=2)
    cfg["model"].update(overrides)
    with _record_lightgbm_calls() as seen:
        Model(cfg, data=_df_for(task)).fit()
    return seen


@pytest.mark.parametrize("task", TASKS)
def test_train_param_names_are_lightgbm_names(task: str) -> None:
    """Every key handed to ``lgb.train`` must be one LightGBM defines."""
    seen = _fit_with_every_smart_parameter(task)
    assert seen["train_params"], "no lgb.train call was recorded"
    keys = sorted({k for params in seen["train_params"] for k in params})
    unknown = [k for k in keys if k not in LGBM_NAMES]
    assert not unknown, (
        f"keys handed to lgb.train that LightGBM does not define: {unknown}. "
        f"LightGBM {lgb.__version__} defines {len(CANONICAL)} names "
        f"({len(LGBM_NAMES)} including aliases). An undefined key is discarded "
        "silently, so whatever it was meant to do never happened."
    )


@pytest.mark.parametrize("task", TASKS)
def test_dataset_keywords_are_real_keywords(task: str) -> None:
    """Every keyword handed to ``lgb.Dataset`` must exist in its signature."""
    seen = _fit_with_every_smart_parameter(task)
    assert seen["dataset_kwargs"], "no lgb.Dataset call was recorded"
    keys = sorted({k for kwargs in seen["dataset_kwargs"] for k in kwargs})
    unknown = [k for k in keys if k not in DATASET_KWARGS]
    assert not unknown, (
        f"keywords handed to lgb.Dataset that it does not accept: {unknown}. "
        f"Accepted: {sorted(DATASET_KWARGS)}"
    )


@pytest.mark.parametrize("task", TASKS)
def test_model_params_names_are_lightgbm_names(task: str) -> None:
    """A ``model.params`` key LightGBM does not define must be refused.

    This is the fit-path twin of ``tuning.optuna.space``: whatever is written
    here is forwarded to ``lgb.train`` unchecked, so a typo trains happily and
    changes nothing.
    """
    cfg = make_config(task, n_estimators=5, n_splits=2)
    cfg["model"]["params"]["not_a_lightgbm_parameter"] = 123

    with pytest.raises(LizyMLError) as exc:
        Model(cfg, data=_df_for(task)).fit()
    assert exc.value.code is ErrorCode.CONFIG_INVALID
    assert "not_a_lightgbm_parameter" in str(exc.value), (
        "the rejection must name the offending key so the user can fix it; "
        f"got: {exc.value}"
    )


@pytest.mark.parametrize("task", TASKS)
def test_a_config_mutated_after_construction_is_still_checked(task: str) -> None:
    """The caller keeps a reference to the config; the gate must not be fooled.

    ``Model`` stores the ``LizyMLConfig`` it was handed rather than copying it,
    so anything the caller changes afterwards is what actually reaches the
    estimator. A check that ran once at construction would pass over exactly
    this.
    """
    cfg = load_config(make_config(task, n_estimators=5, n_splits=2))
    model = Model(cfg, data=_df_for(task))
    cfg.model.params["not_a_lightgbm_parameter"] = 123

    with _record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as exc:
        model.fit()
    assert exc.value.code is ErrorCode.CONFIG_INVALID
    forwarded = [p for p in seen["train_params"] if "not_a_lightgbm_parameter" in p]
    assert not forwarded, (
        f"the mutated key still reached lgb.train in {len(forwarded)} call(s)"
    )


@pytest.mark.parametrize("task", TASKS)
def test_restored_tuning_params_are_checked_before_refit(task: str) -> None:
    """``best_model_params`` restored from an artifact are installed post-init.

    ``Model.load()`` puts them on the instance after ``__init__`` has run, so
    they reach the estimator without passing anything a constructor could
    check. An artifact written before this gate existed can carry a name
    LightGBM never honoured.
    """
    model = Model(make_config(task, n_estimators=5, n_splits=2), data=_df_for(task))
    model._tuning_result = TuningResult(
        best_model_params={"not_a_lightgbm_parameter": 7},
        best_smart_params={},
        best_training_params={},
        best_score=0.0,
        metric_name="rmse",
        direction="minimize",
        trials=(),
        rounds=(),
    )

    with _record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as exc:
        model.fit()
    assert exc.value.code is ErrorCode.CONFIG_INVALID
    forwarded = [p for p in seen["train_params"] if "not_a_lightgbm_parameter" in p]
    assert not forwarded, (
        f"the restored key still reached lgb.train in {len(forwarded)} call(s)"
    )


def test_export_code_refuses_a_name_the_generated_script_would_discard(
    tmp_path: Any,
) -> None:
    """The exported ``train.py`` is a fourth route to ``lgb.train``.

    ``build_export_params`` reads the *fitted adapter*, not the config, so
    neither gate on the training path sees those names -- and loading a legacy
    artifact is deliberately permitted, so one written before this gate existed
    can carry a name LightGBM never honoured straight into the generated
    script's ``CFG["lgbm_params"]``.
    """
    model = Model(
        make_config("regression", n_estimators=5, n_splits=2),
        data=make_regression_df(n=120),
    )
    model.fit()
    # `export_code` reads the refit adapter. Stand in for one restored from an
    # artifact written before this gate existed.
    assert model._refit_result is not None
    model._refit_result.model.params["not_a_lightgbm_parameter"] = 7

    with pytest.raises(LizyMLError) as exc:
        model.export_code(tmp_path / "export")
    assert exc.value.code is ErrorCode.CONFIG_INVALID
    assert "not_a_lightgbm_parameter" in str(exc.value)


def test_loading_an_artifact_is_not_blocked_by_the_gate() -> None:
    """A saved artifact must still load even if its config carries a bad name.

    An artifact records a fit that happened. Refusing to read one back because
    it names a parameter LightGBM ignored helps nobody, and would break every
    artifact written before this gate existed (H-0093 decision 4). Training
    from it is a different matter and is refused above.
    """
    cfg = load_config(make_config("regression", n_estimators=5, n_splits=2))
    cfg.model.params["not_a_lightgbm_parameter"] = 123
    # Construction is the operation `Model.load()` performs to rebuild the
    # instance; it must not raise.
    Model(cfg, data=make_regression_df(n=120))


# --------------------------------------------------------------------------
# The codegen templates hand names to LightGBM too
# --------------------------------------------------------------------------
def _template_sources() -> dict[str, str]:
    """Return every codegen template: the module-level string constants.

    Selection is by *shape*, not by content. Filtering on the substring
    ``lgb.`` would silently drop a template that imports LightGBM under its full
    name -- the same shape ``_is_lgb`` claims to accept -- so a template written
    with ``lightgbm.train`` would never be scanned and would never be reported
    as unscanned either (DC1). Anything that parses as Python is admitted;
    whether it calls LightGBM is decided later, by reading it.
    """
    import lizyml.codegen.templates as templates

    out: dict[str, str] = {}
    for name, value in vars(templates).items():
        if name.startswith("__") or not isinstance(value, str):
            continue
        try:
            ast.parse(value)
        except SyntaxError:
            # Not a source template (a docstring, a format fragment); it cannot
            # contain a call site, so it is out of the population by shape.
            continue
        out[name] = value
    return out


#: Receiver names treated as LightGBM even when the source does not import it.
#:
#: A union with the resolved aliases, not a replacement for them. A codegen
#: template can be a fragment whose ``import`` lives in another constant, and a
#: fragment that lost its import would otherwise resolve to no aliases and be
#: reported clean. Being wrong here can only over-report -- a module that binds
#: something else to the name ``lgb`` produces a loud extra finding, never a
#: silent pass.
_CONVENTIONAL_LGB_NAMES: frozenset[str] = frozenset({"lgb", "lightgbm"})


def _lgb_call(node: ast.AST, bindings: LightGBMBindings) -> str | None:
    """Return the LightGBM attribute *node* calls, or ``None``.

    *bindings* comes from the enclosing source's own import statements, so
    ``import lightgbm as lgbm`` is seen. Hardcoding ``{"lgb", "lightgbm"}`` was
    the defect this replaces: ``lizyml/calibration/isotonic.py`` imports
    LightGBM as ``lgbm`` and passed user parameters to ``lgbm.train`` while the
    inventory below reported the tree fully covered (DC1).

    Both call shapes are resolved. ``<module>.train(...)`` walks the receiver
    down to the name at its root, so a submodule-qualified call such as
    ``lgb.basic.Dataset(...)`` is seen too -- comparing only the immediate
    receiver would have dropped it with no report, which is the same
    false-clean the alias fix closes. The comparison at the root is exact,
    because ``endswith`` on the dotted name would also match ``not_lgb.train``
    (DC2), and a root that is not a LightGBM name (``self.lgb.train``) does not
    match. ``train(...)`` compares the bare name against what
    ``from lightgbm import`` bound, and answers with the *original* attribute
    name, so ``from lightgbm import train as t`` is reported as ``train``.
    """
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Attribute):
        root: ast.expr = func.value
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and root.id in (
            bindings.modules | _CONVENTIONAL_LGB_NAMES
        ):
            return func.attr
        return None
    if isinstance(func, ast.Name) and func.id in bindings.attrs:
        return bindings.attrs[func.id]
    return None


def _own_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """Walk *scope* without descending into a nested function or class.

    Each call site must be attributed to the innermost scope that contains it.
    Walking the module into a function body would judge a local name against
    the module's bindings, where the same name may be bound several times --
    and the site would then be reported unreadable although it is perfectly
    readable one level down.
    """
    for child in ast.iter_child_nodes(scope):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        yield child
        yield from _own_nodes(child)


def _dict_literals_in_scope(scope: ast.AST) -> dict[str, ast.Dict]:
    """Map local names bound *exactly once* to a dict literal within *scope*.

    Bound more than once, the value at the call site is not decidable by
    reading, so the name is left out and its call is reported as unresolved
    rather than guessed at.
    """
    counts: dict[str, int] = {}
    literals: dict[str, ast.Dict] = {}
    for node in _own_nodes(scope):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                counts[target.id] = counts.get(target.id, 0) + 1
                if isinstance(node.value, ast.Dict):
                    literals[target.id] = node.value
    return {name: d for name, d in literals.items() if counts.get(name) == 1}


def _codegen_lightgbm_names(
    sources: dict[str, str] | None = None,
) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    """Classify every LightGBM call in the codegen templates.

    Returns ``(named, from_config, unresolved)``. The three are exhaustive, so
    nothing drops out silently (DC1):

    * ``named`` -- ``(kind, name)`` for a name readable here: a string key of
      the dict handed to ``lgb.train``, whether written inline or bound once to
      a local, and each keyword of an ``lgb.Dataset`` call.
    * ``from_config`` -- an ``lgb.train`` whose params are a subscript of
      ``CFG``. Those names come from the exported ``config.json``, written from
      the same config the runtime gate checks, so they are covered there.
    * ``unresolved`` -- anything else. A failure, not a skip: the template grew
      a shape this scan cannot read, so the population stopped being closed.
    """
    named: list[tuple[str, str]] = []
    from_config: list[str] = []
    unresolved: list[str] = []

    for const, source in (sources or _template_sources()).items():
        tree = ast.parse(source)
        bindings = lightgbm_bindings(tree)
        # Every function body, plus the module itself, so that a params dict
        # bound beside its `lgb.train` call is found in the same scope.
        scopes: list[ast.AST] = [tree]
        scopes += [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
        ]
        for scope in scopes:
            literals = _dict_literals_in_scope(scope)
            for node in _own_nodes(scope):
                if _lgb_call(node, bindings) == "Dataset":
                    assert isinstance(node, ast.Call)
                    for kw in node.keywords:
                        if kw.arg is not None:
                            named.append(("Dataset keyword", kw.arg))
                        else:
                            # `lgb.Dataset(**something)`: the keywords are not
                            # visible here. Dropping it would hide whatever it
                            # unpacks, so it is reported (DC1).
                            unresolved.append(
                                f"{const}: lgb.Dataset(**...) hides its keywords"
                            )
                elif _lgb_call(node, bindings) == "train":
                    assert isinstance(node, ast.Call)
                    first = node.args[0] if node.args else None
                    target: ast.AST | None = first
                    if isinstance(first, ast.Name) and first.id in literals:
                        target = literals[first.id]
                    if isinstance(target, ast.Dict):
                        for key in target.keys:
                            if isinstance(key, ast.Constant) and isinstance(
                                key.value, str
                            ):
                                named.append(("train param", key.value))
                            else:
                                unresolved.append(
                                    f"{const}: non-literal key in an lgb.train "
                                    "params dict"
                                )
                    elif (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "CFG"
                    ):
                        from_config.append(f"{const}: lgb.train(CFG[...])")
                    else:
                        unresolved.append(
                            f"{const}: lgb.train params are "
                            f"{type(target).__name__}, which this scan cannot read"
                        )
    # A function body is walked once as its own scope and again via the module,
    # so the same site is seen twice; the names are a set, not a count.
    return sorted(set(named)), sorted(set(from_config)), sorted(set(unresolved))


CODEGEN_NAMES, CODEGEN_FROM_CONFIG, CODEGEN_UNRESOLVED = _codegen_lightgbm_names()


def test_codegen_templates_are_scanned() -> None:
    """The codegen scan must actually reach names, not merely parse (DC1).

    The exported ``train.py`` hands names to LightGBM exactly as the library
    does, and gets the same silent discard. A scan that found nothing would
    make the assertion below vacuous.
    """
    assert _template_sources(), "no codegen template source was found"
    train_params = [n for kind, n in CODEGEN_NAMES if kind == "train param"]
    dataset_kwargs = [n for kind, n in CODEGEN_NAMES if kind == "Dataset keyword"]
    assert train_params, (
        "no lgb.train parameter name was extracted from the templates; the "
        f"config-driven sites were {CODEGEN_FROM_CONFIG} and the unreadable "
        f"ones {CODEGEN_UNRESOLVED}"
    )
    assert dataset_kwargs, "no lgb.Dataset keyword was extracted from the templates"


# --------------------------------------------------------------------------
# The population of routes, enumerated rather than asserted
# --------------------------------------------------------------------------
#: Every place inside ``lizyml/`` where a parameter dict can cross into the
#: estimator, and what covers it.
#:
#: Two rounds of review each falsified a prose claim about "every route" -- the
#: first found post-construction mutation and restored tuning params, the second
#: found the codegen export path. Both times the answer was another call site
#: and another sentence. This inventory replaces the sentence: the scan below
#: fails when it finds a site not named here, **and** when a name here matches
#: no site, so the claim cannot go stale in either direction.
ESTIMATOR_ROUTES: dict[tuple[str, str], str] = {
    (
        "lizyml/estimators/lgbm/adapter.py",
        "lightgbm.train",
    ): "the model's Booster; its params are self.params, set by the one adapter "
    "construction below",
    (
        "lizyml/estimators/lgbm/adapter.py",
        "lightgbm.Dataset",
    ): "Dataset keywords are API arguments, not tunable parameters; checked "
    "against the constructor signature by test_dataset_keywords_are_real_keywords",
    (
        "lizyml/estimators/lgbm/provider.py",
        "LGBMAdapter",
    ): "the door: the single place a params dict reaches an adapter. Every "
    "producer below is gated before it gets here",
    (
        "lizyml/core/model.py",
        "build_estimator_factory(params=)",
    ): "the only producer, in _fit_impl; the dict is resolved_model, which "
    "_merge_params gates, and the tuning space and export path are gated at "
    "their own call sites",
    (
        "lizyml/calibration/isotonic.py",
        "lightgbm.train",
    ): "the calibrator's Booster; its params are calibration.params merged over "
    "_ISOTONIC_DEFAULTS, gated by check_calibration_param_names in the Facade",
    (
        "lizyml/calibration/isotonic.py",
        "lightgbm.Dataset",
    ): "Dataset keywords here are literals in the calibrator, not user input; "
    "calibration.params never reaches them",
}


def _estimator_routes_in_package(
    sources: dict[str, str] | None = None,
) -> set[tuple[str, str]]:
    """Scan ``lizyml/`` for every site that can hand params to the estimator.

    Three node shapes qualify, and each was added because the previous shape
    set reported a clean tree that was not clean:

    * a direct ``lightgbm.train`` / ``lightgbm.Dataset`` call, under whatever
      name the module binds LightGBM to. Matching ``{"lgb", "lightgbm"}`` by
      hand missed ``calibration/isotonic.py``, which says ``import lightgbm as
      lgbm`` and hands it ``calibration.params``;
    * the construction of an estimator adapter with ``params=`` -- the door
      itself; and
    * a call to ``build_estimator_factory`` with ``params=`` -- a *producer* of
      that dict. Scanning only the door meant a new in-package caller could
      reach the adapter while the inventory stayed green, so "the test fails
      when a route appears" was not true of the producers.

    The codegen templates are not included: their calls live inside string
    constants, so they are not calls in this module's AST, and they have their
    own scan above.
    """
    if sources is None:
        sources = {
            path.relative_to(REPO).as_posix(): path.read_text(encoding="utf-8")
            for path in sorted((REPO / "lizyml").rglob("*.py"))
        }
    found: set[tuple[str, str]] = set()
    for rel, source in sources.items():
        tree = ast.parse(source)
        bindings = lightgbm_bindings(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            attr = _lgb_call(node, bindings)
            if attr in ("train", "Dataset"):
                # Recorded under the canonical name, not the local alias: the
                # route is the same one however the importing module spells it.
                found.add((rel, f"lightgbm.{attr}"))
            func = node.func
            takes_params = any(kw.arg == "params" for kw in node.keywords)
            if isinstance(func, ast.Name) and func.id.endswith("Adapter"):
                if takes_params:
                    found.add((rel, func.id))
            elif takes_params and _called_name(func) == "build_estimator_factory":
                found.add((rel, "build_estimator_factory(params=)"))
    return found


def _called_name(func: ast.expr) -> str | None:
    """The final name of a call target: ``a.b.c()`` -> ``c``, ``f()`` -> ``f``."""
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


#: Sources the scan must classify correctly. Each is a route that could be
#: added to ``lizyml/`` tomorrow; the value is what the scan must report.
#:
#: These exist because the inventory's guarantee -- "a new route fails this
#: test on the commit that introduces it" -- was asserted rather than
#: exercised, and was false: the scan looked only at the adapter construction,
#: so a new caller of ``build_estimator_factory`` produced no detection at all.
HOSTILE_ROUTE_SHAPES: dict[str, tuple[str, set[tuple[str, str]]]] = {
    "aliased import": (
        "import lightgbm as lgbm\ndef f(p, d):\n    return lgbm.train(p, d)\n",
        {("m.py", "lightgbm.train")},
    ),
    "from-import": (
        "from lightgbm import train\ndef f(p, d):\n    return train(p, d)\n",
        {("m.py", "lightgbm.train")},
    ),
    "from-import renamed": (
        "from lightgbm import train as t\ndef f(p, d):\n    return t(p, d)\n",
        {("m.py", "lightgbm.train")},
    ),
    "new factory producer": (
        "def f(provider, p):\n"
        "    return provider.build_estimator_factory(task='binary', params=p)\n",
        {("m.py", "build_estimator_factory(params=)")},
    ),
    "bare factory producer": (
        "def f(p):\n    return build_estimator_factory(params=p)\n",
        {("m.py", "build_estimator_factory(params=)")},
    ),
    "new adapter construction": (
        "def f(p):\n    return SomeAdapter(params=p)\n",
        {("m.py", "SomeAdapter")},
    ),
    "submodule-qualified": (
        "import lightgbm as lgb\n"
        "def f(X, y):\n"
        "    return lgb.basic.Dataset(X, label=y)\n",
        {("m.py", "lightgbm.Dataset")},
    ),
    "attribute root is not a module (negative control)": (
        "class C:\n    def f(self, p, d):\n        return self.lgb.train(p, d)\n",
        set(),
    ),
    "not lightgbm (negative control)": (
        "import not_lgb\ndef f(p, d):\n    return not_lgb.train(p, d)\n",
        set(),
    ),
    "factory without params (negative control)": (
        "def f(provider):\n"
        "    return provider.build_estimator_factory(task='binary')\n",
        set(),
    ),
}


@pytest.mark.parametrize("label", sorted(HOSTILE_ROUTE_SHAPES))
def test_the_scan_detects_an_injected_route(label: str) -> None:
    """The inventory's value is that a new route cannot arrive quietly.

    Asserting that in prose is what let two of these shapes through. Each case
    is a source the scan is fed directly, so the claim is exercised rather than
    described -- including two negative controls, because a scan that reports
    everything would pass the positive cases and be useless.
    """
    source, expected = HOSTILE_ROUTE_SHAPES[label]
    assert _estimator_routes_in_package({"m.py": source}) == expected


def test_every_route_into_the_estimator_is_inventoried() -> None:
    """The set of routes must be enumerated, not claimed.

    A gate whose coverage is stated in prose is only as good as the last
    reviewer who went looking. This makes the population a checked object: a new
    way to put a parameter dict in front of LightGBM fails here, on the commit
    that introduces it, rather than at the next review.

    **What this does and does not claim.** It is a tripwire over the call
    shapes in ``HOSTILE_ROUTE_SHAPES`` -- the shapes this package actually
    writes, plus the ones review has produced. Python's call grammar is open
    (``getattr(lgb, "train")``, ``functools.partial``, a dispatch table), and no
    AST walker closes it; a scan that chased each new shape would grow without
    bound while the gate it protects did not. **That the shipped code is covered
    today is not this test's evidence.** It rests on the gates sitting at the
    points of use, on the measured firing rates in H-0093, and on
    ``test_train_param_names_are_lightgbm_names``, which reads what a real fit
    handed to a real ``lgb.train``. This test's job is narrower and worth
    having: it makes tomorrow's ordinary new caller loud instead of silent.
    """
    found = _estimator_routes_in_package()
    declared = set(ESTIMATOR_ROUTES)
    unlisted = sorted(found - declared)
    stale = sorted(declared - found)
    assert not unlisted, (
        f"these sites can hand parameters to the estimator and are not in "
        f"ESTIMATOR_ROUTES: {unlisted}. Add each with the gate that covers it, "
        "or gate it."
    )
    assert not stale, (
        f"ESTIMATOR_ROUTES names sites that no longer exist: {stale}. An entry "
        "matching nothing describes a route that is not there, which makes the "
        "inventory read as wider coverage than it has."
    )
    assert found, "the scan found no route at all; it is not looking correctly"


#: Shapes the extractor must not lose. Each was a real silent drop before it
#: was closed: a template written against the full module name was filtered out
#: by a substring test, and `**kwargs` unpacking was skipped rather than
#: reported. The value is what the extractor must say about the shape.
HOSTILE_TEMPLATE_SHAPES: dict[str, tuple[str, str]] = {
    "full module name": (
        'import lightgbm\nlightgbm.train({"not_a_lightgbm_parameter": 1}, None)\n',
        "named",
    ),
    "Dataset kwargs unpacking": (
        'import lightgbm as lgb\nlgb.Dataset([], **{"not_a_dataset_keyword": 1})\n',
        "unresolved",
    ),
    "train params from a call": (
        "import lightgbm as lgb\nlgb.train(build_params(), None)\n",
        "unresolved",
    ),
    "params rebound twice": (
        "import lightgbm as lgb\n"
        "def f():\n"
        '    params = {"num_leaves": 3}\n'
        '    params = {"max_depth": 2}\n'
        "    lgb.train(params, None)\n",
        "unresolved",
    ),
}


@pytest.mark.parametrize(
    ("shape", "source", "expected"),
    [(k, v[0], v[1]) for k, v in HOSTILE_TEMPLATE_SHAPES.items()],
    ids=list(HOSTILE_TEMPLATE_SHAPES),
)
def test_the_extractor_loses_no_call_shape(
    shape: str, source: str, expected: str
) -> None:
    """Every shape must land in one of the three buckets, never in none.

    Exhaustiveness is the whole claim of the classification, and a shape that
    falls through every branch produces an empty delta in all three -- which
    reads exactly like a clean template (DC1). These are the shapes that did
    fall through.
    """
    named, from_config, unresolved = _codegen_lightgbm_names({shape: source})
    buckets = {
        "named": named,
        "from_config": from_config,
        "unresolved": unresolved,
    }
    assert buckets[expected], (
        f"{shape!r} produced nothing in {expected!r}; the extractor saw "
        f"named={named}, from_config={from_config}, unresolved={unresolved}"
    )
    assert any(buckets.values()), f"{shape!r} fell through every bucket"


def test_a_clean_template_produces_no_findings() -> None:
    """The negative control: a valid shape must not be reported unresolved.

    Without this, an extractor that called everything unresolved would satisfy
    every assertion above.
    """
    named, from_config, unresolved = _codegen_lightgbm_names(
        {
            "clean": "import lightgbm as lgb\n"
            'lgb.train({"num_leaves": 3}, lgb.Dataset([], label=[]))\n'
        }
    )
    assert not unresolved, unresolved
    assert ("train param", "num_leaves") in named
    assert ("Dataset keyword", "label") in named
    assert not from_config


def test_every_codegen_call_site_is_classified() -> None:
    """No LightGBM call site in the templates may be left unreadable.

    The three outcomes -- a name read here, params taken from the exported
    config, or unreadable -- are exhaustive by construction. An unreadable site
    means the template grew a shape this scan cannot follow, and the population
    silently stopped being closed (DC1). Reported as a failure, never skipped.
    """
    assert not CODEGEN_UNRESOLVED, (
        f"these codegen LightGBM call sites could not be read: "
        f"{CODEGEN_UNRESOLVED}. Extend the scan to the new shape, or the names "
        "at those sites go unchecked."
    )
    assert CODEGEN_FROM_CONFIG, (
        "no lgb.train(CFG[...]) site was found. The generated script is "
        "expected to take its training params from the exported config; if "
        "that stopped being true, the runtime gate no longer covers them."
    )


def test_codegen_template_names_are_lightgbm_names() -> None:
    """Every name the generated code hands to LightGBM must be a real one."""
    unknown = [
        (kind, name)
        for kind, name in CODEGEN_NAMES
        if name not in (LGBM_NAMES if kind == "train param" else DATASET_KWARGS)
    ]
    assert not unknown, (
        f"the codegen templates hand LightGBM names it does not define: "
        f"{unknown}. The exported script gets the same silent discard the "
        "library gives any unknown key."
    )
