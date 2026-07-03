"""Shared optional-dependency guard for plot modules (#218).

Previously each plot module defined its own ``_require_plotly`` / ``_check_plotly``
with a slightly different message string. This centralizes the check so the
``OPTIONAL_DEP_MISSING`` contract and message format are declared once.
"""

from __future__ import annotations

from typing import Any

from lizyml.core.exceptions import ErrorCode, LizyMLError


def require_plotly(plotly_module: Any, *, feature: str = "plots") -> None:
    """Raise ``LizyMLError(OPTIONAL_DEP_MISSING)`` when plotly is unavailable.

    Args:
        plotly_module: The module's ``_plotly`` sentinel (``None`` when the
            optional ``plotly`` import failed).
        feature: Human-readable name of the plotting feature, used in the
            message (e.g. ``"calibration plots"``).
    """
    if plotly_module is None:
        raise LizyMLError(
            code=ErrorCode.OPTIONAL_DEP_MISSING,
            user_message=(
                f"plotly is required for {feature}. "
                "Install with: pip install 'lizyml[plots]'"
            ),
            context={"package": "plotly"},
        )
