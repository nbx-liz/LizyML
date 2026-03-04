"""ExportSpec: export configuration (reserved for future use)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExportSpec:
    """Normalized export configuration."""

    path: Path | None = None
    format_version: int = 1
