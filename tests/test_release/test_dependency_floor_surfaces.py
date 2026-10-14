"""Places that restate a dependency floor agree with ``pyproject.toml`` (H-0126).

``pyproject.toml`` is the one declaration. The generated project's
``requirements.txt`` and the adapter's install hint repeat the LightGBM floor,
and the ``dev`` dependency group must take the extras' floors by reference
instead of declaring them a second time.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement

from lizyml.codegen.templates import render_requirements_txt
from lizyml.estimators.lgbm import adapter

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10; pytest itself depends on tomli there
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]


def _pyproject() -> dict[str, Any]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _lightgbm_floor() -> str:
    for text in _pyproject()["project"]["dependencies"]:
        req = Requirement(text)
        if req.name == "lightgbm":
            (spec,) = req.specifier
            assert spec.operator == ">="
            return spec.version
    raise AssertionError("pyproject.toml declares no lightgbm requirement")


def test_generated_requirements_carry_the_declared_lightgbm_floor() -> None:
    floor = _lightgbm_floor()
    for uses_scipy in (False, True):
        lines = render_requirements_txt(uses_scipy=uses_scipy).splitlines()
        assert [line for line in lines if line.startswith("lightgbm")] == [
            f"lightgbm>={floor}"
        ]


def test_the_install_hint_names_the_declared_lightgbm_floor() -> None:
    assert f'pip install "lightgbm>={_lightgbm_floor()}"' in adapter._INSTALL_HINT


def test_the_dev_group_takes_the_extras_by_reference() -> None:
    data = _pyproject()
    extras = data["project"]["optional-dependencies"]
    extra_packages = {Requirement(t).name for reqs in extras.values() for t in reqs}
    dev = [
        Requirement(t) for t in data["dependency-groups"]["dev"] if isinstance(t, str)
    ]
    assert not {req.name for req in dev} & extra_packages
    (self_ref,) = [req for req in dev if req.name == "lizyml"]
    assert self_ref.extras == set(extras)
