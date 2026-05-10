"""Common plot layout / theme helpers (#123).

Centralises the ``fig.update_layout(...)`` call so a future theme change
(template, brand colors, fonts, dark mode) can be made in one place rather
than across every plot module.

Public helper:

    apply_default_layout(fig, *, title, **layout)

Each plot module routes its layout call through this helper. Existing
per-plot kwargs (``height``, ``width``, ``xaxis_title``, ``barmode`` ...)
remain plot-specific data and are forwarded as ``**layout``.
"""

from __future__ import annotations

from typing import Any

DEFAULT_TEMPLATE = "plotly"
"""Plotly template applied to every LizyML plot. Centralised here to make
future theme switches a one-line change."""

DEFAULT_HEIGHT = 400
"""Default height in pixels when a plot does not pick its own."""

DEFAULT_WIDTH = 600
"""Default width in pixels when a plot does not pick its own."""


def apply_default_layout(
    fig: Any,
    *,
    title: str,
    **layout: Any,
) -> None:
    """Apply LizyML's default layout to ``fig`` in place.

    Args:
        fig: A plotly ``Figure`` (typed as ``Any`` so this module does not
            require plotly at import time).
        title: Figure title — required so every plot has one.
        **layout: Forwarded to ``fig.update_layout``. Recognised keys
            include ``height``, ``width``, ``xaxis_title``, ``yaxis_title``,
            ``barmode``, ``margin``, etc. Unset ``height`` / ``width``
            inherit ``DEFAULT_HEIGHT`` / ``DEFAULT_WIDTH``.

    Notes:
        Pass ``height=None`` / ``width=None`` explicitly to omit a default
        (rare — only the multiclass OOF distribution and one residuals
        sub-plot currently rely on plotly's auto-sizing).
    """
    final_layout: dict[str, Any] = {
        "template": DEFAULT_TEMPLATE,
        "title": title,
    }
    if "height" not in layout:
        final_layout["height"] = DEFAULT_HEIGHT
    if "width" not in layout:
        final_layout["width"] = DEFAULT_WIDTH
    # Only ``None`` is filtered out so plotly's auto-sizing kicks in for that
    # key. Falsy-but-meaningful values (``0``, ``False``, ``""``) are valid
    # plotly inputs and pass through untouched.
    final_layout.update({k: v for k, v in layout.items() if v is not None})
    fig.update_layout(**final_layout)
