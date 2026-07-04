"""Coverage for LizyMLError.__repr__ (all fields / minimal)."""

from __future__ import annotations

from lizyml.core.exceptions import ErrorCode, LizyMLError


class TestLizyMLErrorRepr:
    def test_repr_with_all_fields(self) -> None:
        err = LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message="bad config",
            debug_message="detail",
            context={"key": "val"},
            cause=ValueError("root"),
        )
        r = repr(err)
        assert "debug_message='detail'" in r
        assert "context={'key': 'val'}" in r
        assert "cause=" in r

    def test_repr_minimal(self) -> None:
        err = LizyMLError(code=ErrorCode.CONFIG_INVALID, user_message="bad")
        r = repr(err)
        assert "debug_message" not in r
        assert "context" not in r
        assert "cause" not in r
