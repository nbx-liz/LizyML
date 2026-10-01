"""``config_version`` is enforced on every path into ``Model`` (H-0106, #272).

The check used to live only in ``load_config``. A ``LizyMLConfig`` built with
``model_validate`` (or ``model_construct``, or mutated, or copied with an
update), an environment override applied after the check, and
``config_version: false`` (coerced to ``0``) all reached ``Model`` unchecked.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from lizyml.config import loader
from lizyml.config.loader import load_config
from lizyml.config.schema import LizyMLConfig
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.model import Model
from tests._helpers import make_config


def _raw(v: Any) -> dict[str, Any]:
    cfg = make_config("binary")
    cfg["config_version"] = v
    return cfg


def _assigned(v: Any) -> LizyMLConfig:
    cfg = LizyMLConfig.model_validate(_raw(1))
    cfg.config_version = v
    return cfg


#: entry path -> builds a Model (or a config) from a requested version.
_ENTRIES: dict[str, Callable[[Any, pytest.MonkeyPatch], object]] = {
    "load_config(dict)": lambda v, mp: load_config(_raw(v)),
    "Model(dict)": lambda v, mp: Model(_raw(v)),
    "model_validate": lambda v, mp: LizyMLConfig.model_validate(_raw(v)),
    "Model(model_validate)": lambda v, mp: Model(LizyMLConfig.model_validate(_raw(v))),
    "Model(model_construct)": lambda v, mp: Model(
        LizyMLConfig.model_construct(**_raw(v))
    ),
    "Model(assigned)": lambda v, mp: Model(_assigned(v)),
    "Model(model_copy)": lambda v, mp: Model(
        LizyMLConfig.model_validate(_raw(1)).model_copy(update={"config_version": v})
    ),
    "env override": lambda v, mp: (
        mp.setenv("LIZYML__config_version", str(v)),
        Model(_raw(1)),
    ),
}


@pytest.mark.parametrize("entry", sorted(_ENTRIES))
@pytest.mark.parametrize("v", [1, 2])
def test_entry_path_by_version(
    entry: str, v: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = _ENTRIES[entry]
    if v in loader.SUPPORTED_CONFIG_VERSIONS:
        build(v, monkeypatch)
        return
    with pytest.raises(LizyMLError) as exc:
        build(v, monkeypatch)
    assert exc.value.code == ErrorCode.CONFIG_VERSION_UNSUPPORTED
    assert exc.value.context["supported"] == loader.SUPPORTED_CONFIG_VERSIONS


@pytest.mark.parametrize(
    "entry",
    [
        "load_config(dict)",
        "Model(dict)",
        "model_validate",
        "Model(model_validate)",
        "env override",
    ],
)
def test_false_is_not_version_zero(entry: str, monkeypatch: pytest.MonkeyPatch) -> None:
    value: Any = "false" if entry == "env override" else False
    with pytest.raises(LizyMLError) as exc:
        _ENTRIES[entry](value, monkeypatch)
    assert exc.value.code == ErrorCode.CONFIG_VERSION_UNSUPPORTED


@pytest.mark.parametrize("value", ["2", "2.0", " 2 ", 2.5, -1.5, 0.5])
def test_loader_context_keeps_the_raw_value(value: object) -> None:
    """The loader refuses on the value as written, so the context keeps it.
    Fractional floats are truncated there, as before H-0106 (2.5 -> 2)."""
    with pytest.raises(LizyMLError) as exc:
        load_config(_raw(value))
    assert exc.value.code == ErrorCode.CONFIG_VERSION_UNSUPPORTED
    assert exc.value.context["config_version"] == value


@pytest.mark.parametrize("value", [1.5, "1.5", "x", None, float("inf"), float("nan")])
def test_values_that_are_not_versions_are_config_invalid(value: object) -> None:
    """Not a version at all: pydantic's type error, CONFIG_INVALID, as before
    (``inf`` used to escape as a raw ``OverflowError``)."""
    with pytest.raises(LizyMLError) as exc:
        load_config(_raw(value))
    assert exc.value.code == ErrorCode.CONFIG_INVALID


@pytest.mark.parametrize("value", [1.5, "x", None])
def test_unvalidated_instance_with_a_non_version_is_refused(value: object) -> None:
    """No pydantic step follows Model's own check, so it refuses outright and
    does not truncate 1.5 to version 1."""
    with pytest.raises(LizyMLError) as exc:
        Model(_assigned(value))
    assert exc.value.code == ErrorCode.CONFIG_VERSION_UNSUPPORTED


def test_supported_versions_is_one_object() -> None:
    from lizyml.config import version

    assert loader.SUPPORTED_CONFIG_VERSIONS is version.SUPPORTED_CONFIG_VERSIONS


_UNVALIDATED = ["Model(model_construct)", "Model(assigned)", "Model(model_copy)"]


@pytest.mark.parametrize("entry", _UNVALIDATED)
def test_false_is_refused_when_model_receives_it(
    entry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """These paths skip validation and keep the ``bool``. Building the config
    succeeds; ``Model`` refuses it on receipt."""
    with pytest.raises(LizyMLError) as exc:
        _ENTRIES[entry](False, monkeypatch)
    assert exc.value.code == ErrorCode.CONFIG_VERSION_UNSUPPORTED


@pytest.mark.parametrize("entry", sorted(_ENTRIES))
@pytest.mark.parametrize("value", [True, "1"])
def test_true_and_string_one_are_version_one(
    entry: str, value: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``int()`` conversion on every path: ``True`` and ``"1"`` are version 1."""
    _ENTRIES[entry](
        str(value).lower() if entry == "env override" else value, monkeypatch
    )
