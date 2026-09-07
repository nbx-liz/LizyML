"""Resolve which local names a module binds to LightGBM.

Two tests scan ``lizyml/`` for code that can hand a parameter dict to LightGBM:
``test_estimators/test_lightgbm_parameter_names.py`` enumerates every such
route, and ``test_calibration/test_calibration_param_names.py`` checks that
every LightGBM-backed calibrator is declared as one. Both need the same answer
to the same question, and getting it wrong is not a loud failure -- a scan that
misses an alias reports a clean tree.

That is not hypothetical. Both scans originally hardcoded ``{"lgb",
"lightgbm"}``; ``lizyml/calibration/isotonic.py`` says ``import lightgbm as
lgbm`` and was invisible to them, and its ``lgbm.train`` call took
user-supplied parameters with nothing checking their names. Reading each
module's own imports is what closes that, so it lives in one place.
"""

from __future__ import annotations

import ast
from typing import NamedTuple


class LightGBMBindings(NamedTuple):
    """What a module's own imports bind to LightGBM.

    The two are kept apart because they are called differently and conflating
    them invents routes: ``modules`` are called as ``<name>.train(...)``, while
    ``attrs`` are already the function, called as ``<name>(...)``.
    """

    #: Local names bound to the ``lightgbm`` module itself.
    modules: frozenset[str]
    #: Local name -> the ``lightgbm`` attribute it was imported from.
    attrs: dict[str, str]


def lightgbm_bindings(tree: ast.AST) -> LightGBMBindings:
    """Return the LightGBM names *tree* binds, from its own import statements.

    Covers the three import forms Python offers: ``import lightgbm``,
    ``import lightgbm as X`` (including a submodule, which binds the package
    root without an ``as``), and ``from lightgbm[.sub] import Y [as Z]``.
    """
    modules: set[str] = set()
    attrs: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "lightgbm" or alias.name.startswith("lightgbm."):
                    modules.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root != "lightgbm" or node.level:
                continue
            for alias in node.names:
                attrs[alias.asname or alias.name] = alias.name
    return LightGBMBindings(frozenset(modules), attrs)
