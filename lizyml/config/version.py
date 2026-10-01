"""The supported ``config_version`` values and the one check (H-0106).

The check used to live only in ``load_config``, so a ``LizyMLConfig`` built
another way -- ``model_validate``, ``model_construct``, assignment,
``model_copy(update=...)`` -- or a version set by an environment override after
the loader had checked, reached ``Model`` unchecked (#272). Every entry point
now calls :func:`check_config_version`: the schema's field validator, the
loader's raw-value check, and ``Model.__init__`` for instances.
"""

from __future__ import annotations

from lizyml.core.exceptions import ErrorCode, LizyMLError

SUPPORTED_CONFIG_VERSIONS: list[int] = [1]


def _as_version(value: object) -> int | None:
    """Return *value* as an integer version, or ``None`` when it is not one.

    ``bool`` converts like ``int`` (``False`` -> 0, ``True`` -> 1), so a
    ``False`` is refused whether pydantic has already coerced it to ``0`` or an
    unvalidated instance still holds the ``bool``. A float converts only when it
    is integral, and a string only when ``int()`` parses it.
    """
    if isinstance(value, int):  # includes bool
        return int(value)
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def check_config_version(value: object, *, allow_unparsed: bool = False) -> None:
    """Raise ``CONFIG_VERSION_UNSUPPORTED`` unless *value* is a supported version.

    Args:
        value: The ``config_version`` as written or as held by a config.
        allow_unparsed: Return silently when *value* is not a version at all
            (missing, ``1.5``, ``"x"``). The loader passes ``True`` so pydantic
            reports the type error as ``CONFIG_INVALID``, as it always has.
            Every other caller has no later type check, so it refuses.

    Raises:
        LizyMLError: With ``CONFIG_VERSION_UNSUPPORTED``.
    """
    version = _as_version(value)
    if version is None and allow_unparsed:
        return
    if version is not None and version in SUPPORTED_CONFIG_VERSIONS:
        return
    raise LizyMLError(
        ErrorCode.CONFIG_VERSION_UNSUPPORTED,
        user_message=(
            f"config_version={value!r} is not supported. "
            f"Supported versions: {SUPPORTED_CONFIG_VERSIONS}"
        ),
        context={"config_version": value, "supported": SUPPORTED_CONFIG_VERSIONS},
    )
