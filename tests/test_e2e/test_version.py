"""Version consistency tests.

Verifies that lizyml.__version__ matches the installed package metadata.
"""

from __future__ import annotations

from importlib.metadata import version as pkg_version

import lizyml


class TestVersion:
    def test_version_is_string(self) -> None:
        assert isinstance(lizyml.__version__, str)

    def test_version_not_empty(self) -> None:
        assert len(lizyml.__version__) > 0

    def test_version_matches_package_metadata(self) -> None:
        """__version__ must match the installed package metadata."""
        installed_version = pkg_version("lizyml")
        assert lizyml.__version__ == installed_version

    def test_version_format(self) -> None:
        """Version must be a valid semver-like string (X.Y.Z)."""
        parts = lizyml.__version__.split(".")
        assert len(parts) >= 2, "Version must have at least major.minor"
        assert all(p.isdigit() for p in parts[:2]), "Major and minor must be numeric"
