"""Edge-case coverage for data/fingerprint.py."""

from __future__ import annotations

from lizyml.data.fingerprint import DataFingerprint, _hash_file


class TestFingerprint:
    def test_matches_no_file_hash(self) -> None:
        fp1 = DataFingerprint(row_count=10, column_hash="abc")
        fp2 = DataFingerprint(row_count=10, column_hash="abc")
        assert fp1.matches(fp2)

    def test_hash_file_nonexistent(self) -> None:
        assert _hash_file("/nonexistent/path/to/file.csv") is None
