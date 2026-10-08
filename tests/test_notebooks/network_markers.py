"""Strings that mark a notebook failure as a remote-data fetch failure.

Some tutorials fetch a remote dataset (e.g. OpenML credit-g). When the runner
cannot reach the network, the ``CellExecutionError`` text carries one of these.
Shared by the slow execution test (``test_notebook_execution.py``) and the
index execution's retry policy (``index_runner.py``, H-0119 6.(b)).
"""

from __future__ import annotations

NETWORK_ERROR_MARKERS: tuple[str, ...] = (
    "HTTPError",
    "URLError",
    "OpenMLError",
    "api.openml.org",
    "Max retries",
    "ConnectionError",
    "Temporary failure in name resolution",
    "network error",
)
