"""Utilities for importing optional (heavy) dependencies."""

from __future__ import annotations

import importlib
import types

from lizyml.core.exceptions import ErrorCode, LizyMLError


def import_optional(
    module_name: str,
    *,
    package_name: str | None = None,
    install_hint: str | None = None,
) -> types.ModuleType:
    """Import an optional dependency, raising a unified error if missing.

    Args:
        module_name: The Python module to import (e.g. ``"shap"``).
        package_name: The pip package name if different from *module_name*
                      (e.g. ``"scikit-learn"`` for ``"sklearn"``).
        install_hint: Custom installation instruction shown to users.
                      Defaults to ``"pip install <package_name>"``.

    Returns:
        The imported module.

    Raises:
        LizyMLError: With code ``OPTIONAL_DEP_MISSING`` when the module
                     is not installed.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        pkg = package_name or module_name
        hint = install_hint or f"pip install {pkg}"
        raise LizyMLError(
            ErrorCode.OPTIONAL_DEP_MISSING,
            user_message=(
                f"Optional dependency '{module_name}' is not installed. "
                f"Install it with: {hint}"
            ),
            debug_message=str(exc),
            cause=exc,
            context={"module": module_name, "package": pkg},
        ) from exc
