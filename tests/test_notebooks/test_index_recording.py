"""The call recorder and the network guard of the index execution.

H-0119 6.(b) acceptance 7 (the recorder, on ``Model`` directly, no notebook),
plus the runtime replays of acceptance 3 and 6: an instrumented notebook is
executed cell by cell with ``exec`` (fast) or in a real kernel (``slow``), and
each counterexample must fail the recording.

H-0137 replaced acceptance 8 (the retry of network failures) with the network
guard (``network_guard.py``): its acceptance 2 and 3 are the guard cases below,
each with a twin that reads a local file and passes.
"""

from __future__ import annotations

import _io
import _socket
import builtins
import copy
import functools
import inspect
import io
import pathlib
import re
import socket
import sys
import urllib.request
from collections.abc import Callable
from typing import Any

import pytest
from nbclient.exceptions import CellExecutionError

from lizyml import Model
from lizyml.core.exceptions import LizyMLError
from tests._helpers import make_config, make_regression_df
from tests.test_notebooks import index_runner, network_guard
from tests.test_notebooks.index_recorder import CallRecorder


def _public(cls: type) -> dict[str, Any]:
    return {
        name: inspect.getattr_static(cls, name)
        for name, value in inspect.getmembers(cls)
        if not name.startswith("_") and callable(value)
    }


@pytest.fixture(scope="module")
def fitted() -> Model:
    model = Model(make_config("regression", n_estimators=5))
    model.fit(data=make_regression_df(n=120))
    return model


@pytest.fixture
def recorder() -> Any:
    rec = CallRecorder(Model)
    rec.install()
    try:
        yield rec
    finally:
        rec.uninstall()


# --- 7. The recorder ------------------------------------------------------------


def test_load_stays_a_classmethod_bound_alike_from_class_and_instance(
    recorder: CallRecorder, fitted: Model, tmp_path: pathlib.Path
) -> None:
    assert isinstance(inspect.getattr_static(Model, "load"), classmethod)
    assert Model.load.__self__ is Model
    assert fitted.load.__self__ is Model
    fitted.export(tmp_path / "artifact")
    recorder.calls.clear()
    assert isinstance(Model.load(tmp_path / "artifact"), Model)
    assert isinstance(fitted.load(tmp_path / "artifact"), Model)
    assert recorder.calls == [("load", id(Model)), ("load", id(Model))]


def test_properties_are_left_as_they_are(recorder: CallRecorder) -> None:
    del recorder
    assert isinstance(inspect.getattr_static(Model, "fit_result"), property)
    assert "fit_result" not in _public(Model)


def test_static_and_class_methods_and_properties_keep_their_binding() -> None:
    class Sample:
        @staticmethod
        def make(x: int) -> int:
            """Static."""
            return x + 1

        @classmethod
        def build(cls, x: int) -> tuple[type, int]:
            return cls, x

        @property
        def value(self) -> int:
            return 7

        def run(self) -> int:
            return self.make(1) + self.build(2)[1]

    original = dict(vars(Sample))
    rec = CallRecorder(Sample)
    rec.install()
    try:
        assert isinstance(inspect.getattr_static(Sample, "make"), staticmethod)
        assert isinstance(inspect.getattr_static(Sample, "build"), classmethod)
        assert inspect.getattr_static(Sample, "value") is original["value"]
        sample = Sample()
        assert Sample.make(1) == sample.make(1) == 2
        assert Sample.build(3) == sample.build(3) == (Sample, 3)
        assert sample.value == 7
        assert inspect.getdoc(Sample.make) == "Static."
        rec.calls.clear()
        assert sample.run() == 4
        assert rec.calls == [("run", id(sample))], "inner calls are not recorded"
        Sample.make(0)
        assert rec.calls[-1] == ("make", None)
    finally:
        rec.uninstall()
    assert dict(vars(Sample)) == original


def test_names_signatures_and_docstrings_are_unchanged() -> None:
    before = {
        name: (
            inspect.signature(getattr(Model, name)),
            inspect.getdoc(getattr(Model, name)),
        )
        for name in _public(Model)
    }
    rec = CallRecorder(Model)
    rec.install()
    try:
        after = {
            name: (
                inspect.signature(getattr(Model, name)),
                inspect.getdoc(getattr(Model, name)),
            )
            for name in _public(Model)
        }
        assert {n: getattr(Model, n).__name__ for n in after} == {n: n for n in after}
    finally:
        rec.uninstall()
    assert after == before
    assert len(before) >= 20


def test_a_public_method_called_from_another_is_not_recorded(
    recorder: CallRecorder, fitted: Model
) -> None:
    recorder.calls.clear()
    fitted.importance_plot(kind="shap")
    assert recorder.calls == [("importance_plot", id(fitted))]


def test_depth_survives_a_call_that_raises(recorder: CallRecorder) -> None:
    unfitted = Model(make_config("regression"))
    recorder.calls.clear()
    with pytest.raises(LizyMLError):
        unfitted.importance_plot(kind="shap")  # raises inside importance()
    with pytest.raises(LizyMLError):
        unfitted.importance()
    assert recorder.calls == [
        ("importance_plot", id(unfitted)),
        ("importance", id(unfitted)),
    ]


def test_uninstall_restores_the_original_methods() -> None:
    before = _public(Model)
    own = dict(vars(Model))
    rec = CallRecorder(Model)
    rec.install()
    assert _public(Model) != before
    with pytest.raises(RuntimeError, match="installed"):
        rec.install()
    rec.uninstall()
    assert _public(Model) == before
    assert dict(vars(Model)) == own
    with pytest.raises(RuntimeError, match="not installed"):
        rec.uninstall()


def test_expect_confirm_finish(recorder: CallRecorder, fitted: Model) -> None:
    recorder.expect(fitted, "params_table", "1:0")
    fitted.params_table()
    recorder.confirm("1:0")
    assert recorder.finish(["1:0"]) == f"{index_runner.SENTINEL} 1"


@pytest.mark.parametrize(
    "case", ["not a Model", "not called", "another method", "nested expect", "missing"]
)
def test_the_recording_fails(recorder: CallRecorder, fitted: Model, case: str) -> None:
    with pytest.raises(AssertionError):
        if case == "not a Model":
            recorder.expect(object(), "params_table", "1:0")
        elif case == "not called":
            recorder.expect(fitted, "params_table", "1:0")
            recorder.confirm("1:0")
        elif case == "another method":
            recorder.expect(fitted, "params_table", "1:0")
            fitted.evaluate_table()
            recorder.confirm("1:0")
        elif case == "nested expect":
            recorder.expect(fitted, "params_table", "1:0")
            recorder.expect(fitted, "params_table", "1:1")
        else:
            recorder.finish(["1:0"])


# --- Runtime replays (acceptance 3 and 6) ---------------------------------------------


def _cell(
    source: str, *, tagged: bool = False, tags: list[str] | None = None
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if tagged or tags:
        metadata["tags"] = (tags or []) + (["index-example"] if tagged else [])
    return {
        "cell_type": "code",
        "metadata": metadata,
        "source": source,
        "outputs": [],
        "execution_count": None,
    }


SETUP = (
    "from lizyml import Model\n"
    "from tests._helpers import make_config, make_regression_df\n"
    "model = Model(make_config('regression', n_estimators=5))\n"
    "model.fit(data=make_regression_df(n=120))"
)
DECLARED = {"models": ["model"], "methods": ["params_table"], "extras": []}


def _notebook(*cells: dict[str, Any]) -> dict[str, Any]:
    return {
        "cells": list(cells),
        "metadata": {"lizyml": {"index": copy.deepcopy(DECLARED)}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _exec(nb: dict[str, Any]) -> str:
    instrumented = index_runner.instrument_notebook(nb)
    namespace: dict[str, Any] = {}
    printed: list[str] = []
    namespace["print"] = lambda *a: printed.append(" ".join(map(str, a)))
    try:
        for cell in instrumented["cells"]:
            exec(compile(cell["source"], "<cell>", "exec"), namespace)
    finally:
        recorder = namespace.get(index_runner.RECORDER_NAME)
        if recorder is not None:
            recorder.uninstall()
        guard = namespace.get(network_guard.GUARD_NAME)
        if guard is not None:
            guard.uninstall()
    return "\n".join(printed)


def test_the_instrumented_notebook_records_its_examples() -> None:
    nb = _notebook(_cell(SETUP), _cell("model.params_table()", tagged=True))
    assert _exec(nb) == f"{index_runner.SENTINEL} 1"


LOOK_ALIKE = "class Fake:\n    def params_table(self):\n        return 1\n"
RUNTIME_REPLAYS = {
    "a look-alike receiver": LOOK_ALIKE + "model = Fake()",
    "a reassignment": "model = 1",
    "a `def` rebinding the model": "def model():\n    pass",
    "a `class` rebinding the model": (
        "class model:\n    params_table = staticmethod(lambda: 1)"
    ),
    "a subclass that never runs the method": (
        "class Sub(Model):\n    def params_table(self):\n        return 1\n"
        "model = Sub(make_config('regression'))"
    ),
}


@pytest.mark.parametrize("case", sorted(RUNTIME_REPLAYS))
def test_runtime_replays_fail(case: str) -> None:
    nb = _notebook(
        _cell(SETUP),
        _cell(RUNTIME_REPLAYS[case]),
        _cell("model.params_table()", tagged=True),
    )
    with pytest.raises(AssertionError):
        _exec(nb)


def test_instrumentation_does_not_change_the_file(tmp_path: pathlib.Path) -> None:
    nb = _notebook(_cell(SETUP), _cell("model.params_table()", tagged=True))
    snapshot = copy.deepcopy(nb)
    index_runner.instrument_notebook(nb)
    assert nb == snapshot


# --- The network guard (H-0137 acceptance 2 and 3) ------------------------------------

# Never resolvable (RFC 2606), so a broken guard cannot reach a real host either.
HOST = "lizyml-h0137.invalid"
LOOPBACK = ("127.0.0.1", 9)


def _sendmsg() -> object:
    with socket.socket(type=socket.SOCK_DGRAM) as s:
        return s.sendmsg([b"x"], [], 0, LOOPBACK)


# One call per entry point the guard declares (each driven directly, so a hook
# that never fires cannot hide behind an earlier one), quantified over the
# declaration by test_every_declared_entry_point_has_a_call.
ENTRY_POINT_CALLS: dict[str, Callable[[], object]] = {
    # The round-3 finding: the C code resolves a host name inside bind.
    "bind": lambda: socket.socket(type=socket.SOCK_DGRAM).bind((HOST, 0)),
    "connect": lambda: socket.socket().connect(LOOPBACK),
    "connect_ex": lambda: socket.socket().connect_ex(LOOPBACK),
    "sendto": lambda: socket.socket(type=socket.SOCK_DGRAM).sendto(b"x", LOOPBACK),
    "sendmsg": _sendmsg,
    "create_connection": lambda: socket.create_connection((HOST, 443)),
    "getaddrinfo": lambda: socket.getaddrinfo(HOST, 443),
    "gethostbyname": lambda: socket.gethostbyname(HOST),
    "gethostbyname_ex": lambda: socket.gethostbyname_ex(HOST),
    "gethostbyaddr": lambda: socket.gethostbyaddr("127.0.0.1"),
    "getnameinfo": lambda: socket.getnameinfo(LOOPBACK, 0),
    "getfqdn": lambda: socket.getfqdn(HOST),
    "getprotobyname": lambda: socket.getprotobyname("tcp"),
    "getservbyname": lambda: socket.getservbyname("https"),
    "getservbyport": lambda: socket.getservbyport(443),
}
CALLS: dict[str, Callable[[], object]] = {
    **ENTRY_POINT_CALLS,
    "urllib.request.urlopen": lambda: urllib.request.urlopen(f"https://{HOST}/x"),
    # The guard reads no address family: an AF_UNIX path is refused alike. (The
    # method is called unbound because some sandboxes refuse creating an
    # AF_UNIX socket at all.)
    "connect to an AF_UNIX path": lambda: socket.socket.connect(None, "/x.sock"),
    # Functions the table does not refuse because they go through a refused
    # method (DELEGATED below).
    "create_server": lambda: socket.create_server((HOST, 0)),
    "send_fds": lambda: socket.send_fds(
        socket.socket(type=socket.SOCK_DGRAM), [b"x"], [], 0, (HOST, 9)
    ),
}


def test_every_declared_entry_point_has_a_call() -> None:
    declared = (*network_guard.SOCKET_METHODS, *network_guard.MODULE_FUNCTIONS)
    assert sorted(ENTRY_POINT_CALLS) == sorted(declared)


def _public_callables(owner: object) -> set[str]:
    return {
        name
        for name in dir(owner)
        if not name.startswith("_")
        and callable(getattr(owner, name))
        and not inspect.isclass(getattr(owner, name))
    }


@pytest.mark.parametrize(
    ("owner", "refused", "not_refused"),
    [
        (
            socket.socket,
            network_guard.REFUSED_METHODS,
            network_guard.NOT_REFUSED_METHODS,
        ),
        (
            socket,
            network_guard.REFUSED_FUNCTIONS,
            network_guard.NOT_REFUSED_FUNCTIONS,
        ),
    ],
    ids=["socket.socket methods", "socket functions"],
)
def test_the_entry_point_table_classifies_every_public_name(
    owner: object, refused: dict[str, str], not_refused: dict[str, str]
) -> None:
    """A public name Python adds must be classified before the guard passes."""
    assert not set(refused) & set(not_refused)
    assert set(refused) <= _public_callables(owner)
    unclassified = _public_callables(owner) - set(refused) - set(not_refused)
    assert unclassified == set()
    assert all(refused.values()) and all(not_refused.values())


def _documented_parameters(function: object) -> str:
    """The parameters in the first docstring line, ``name(params) -> ...``."""
    first = (inspect.getdoc(function) or "").splitlines()[0]
    match = re.match(r"\w+\((.*?)\)", first)
    return match.group(1) if match else ""


def test_a_method_is_refused_exactly_when_it_takes_an_address() -> None:
    """The method table follows the rule its comment states, read from CPython."""
    takes_address = {
        name
        for name in _public_callables(socket.socket)
        if re.search(
            r"\baddress\b", _documented_parameters(getattr(socket.socket, name))
        )
    }
    assert takes_address == set(network_guard.REFUSED_METHODS)


# Each function the table does not refuse, accounted for in exactly one way.
LOOKUP_EVENTS = frozenset(
    {
        "socket.getaddrinfo",
        "socket.gethostbyaddr",
        "socket.gethostbyname",
        "socket.getnameinfo",
        "socket.getservbyname",
        "socket.getservbyport",
    }
)
_PACKED = b"\x7f\x00\x00\x01"


def _interface(by: str) -> object:
    index, name = socket.if_nameindex()[0]
    return (
        socket.if_indextoname(index) if by == "index" else socket.if_nametoindex(name)
    )


# Called under the guard: no refusal and no lookup audit event.
PROBED: dict[str, Callable[[], object]] = {
    "CMSG_LEN": lambda: socket.CMSG_LEN(1),
    "CMSG_SPACE": lambda: socket.CMSG_SPACE(1),
    "htonl": lambda: socket.htonl(1),
    "htons": lambda: socket.htons(1),
    "ntohl": lambda: socket.ntohl(1),
    "ntohs": lambda: socket.ntohs(1),
    "inet_aton": lambda: socket.inet_aton("127.0.0.1"),
    "inet_ntoa": lambda: socket.inet_ntoa(_PACKED),
    "inet_ntop": lambda: socket.inet_ntop(socket.AF_INET, _PACKED),
    "inet_pton": lambda: socket.inet_pton(socket.AF_INET, "127.0.0.1"),
    "getdefaulttimeout": socket.getdefaulttimeout,
    "setdefaulttimeout": lambda: socket.setdefaulttimeout(socket.getdefaulttimeout()),
    "gethostname": socket.gethostname,
    "has_dualstack_ipv6": socket.has_dualstack_ipv6,
    "if_nameindex": socket.if_nameindex,
    "if_indextoname": lambda: _interface("index"),
    "if_nametoindex": lambda: _interface("name"),
}
# Refused through a method of the table (driven in CALLS).
DELEGATED = {"create_server": "bind", "send_fds": "sendmsg"}
# Not called here, with the reason.
NOT_CALLED = {
    "fromfd": "wraps a file descriptor into a socket.socket, whose methods apply",
    "dup": "duplicates a file descriptor",
    "close": "closes a file descriptor",
    "sethostname": "changes this machine's name and needs privileges",
    "socketpair": "an AF_UNIX pair inside this process; some sandboxes refuse AF_UNIX",
    "recv_fds": "receives on a socket the caller already has; takes no address",
}


def test_each_function_not_refused_is_accounted_for() -> None:
    groups = [set(PROBED), set(DELEGATED), set(NOT_CALLED)]
    assert sum(len(g) for g in groups) == len(set().union(*groups))
    assert set().union(*groups) == set(network_guard.NOT_REFUSED_FUNCTIONS)
    assert set(DELEGATED.values()) <= set(network_guard.REFUSED_METHODS)
    assert set(DELEGATED) <= set(CALLS)


_AUDITED: list[str] | None = None


def _audit(event: str, args: tuple[object, ...]) -> None:
    del args
    if _AUDITED is not None and event.startswith("socket."):
        _AUDITED.append(event)


sys.addaudithook(_audit)  # inert unless _AUDITED is a list


@pytest.mark.parametrize("name", sorted(PROBED))
def test_a_probed_function_looks_nothing_up(name: str, tmp_path: pathlib.Path) -> None:
    global _AUDITED
    guard = network_guard.install(tmp_path)
    _AUDITED = []
    try:
        PROBED[name]()
        events = set(_AUDITED)
    finally:
        _AUDITED = None
        guard.uninstall()
    assert guard.attempts == []
    assert not events & LOOKUP_EVENTS, events


@pytest.mark.parametrize("call", sorted(CALLS))
def test_the_guard_refuses_and_records_each_call(
    call: str, tmp_path: pathlib.Path
) -> None:
    guard = network_guard.install(tmp_path)
    try:
        # create_server re-raises the refusal in its own words; the record and
        # the last-cell check below are what hold for every call.
        match = None if call in DELEGATED else network_guard.MESSAGE
        with pytest.raises(OSError, match=match):
            CALLS[call]()
        hook = call if call in ENTRY_POINT_CALLS else DELEGATED.get(call)
        if hook is not None:
            assert guard.attempts[0].startswith(f"{hook}(")  # this hook fired
        assert len(guard.attempts) == 1
        with pytest.raises(AssertionError, match=network_guard.MESSAGE):
            guard.check()
    finally:
        guard.uninstall()


def _assert_pristine() -> None:
    """The entry points are the interpreter's own, not a leftover replacement.

    Judged against ``_socket`` / ``_io`` rather than a snapshot, which an
    earlier test that failed to uninstall could already have polluted.
    """
    for name in network_guard.SOCKET_METHODS:
        assert getattr(socket.socket, name) is getattr(_socket.socket, name), name
    for name in network_guard.MODULE_FUNCTIONS:
        function = socket.__dict__[name]
        assert function is getattr(_socket, name, None) or (
            function.__module__ == "socket" and function.__name__ == name
        ), name
    assert builtins.open is _io.open and io.open is _io.open


def test_uninstall_restores_every_replaced_entry_point() -> None:
    _assert_pristine()
    guard = network_guard.install()
    assert builtins.open is not _io.open
    assert all(
        getattr(socket.socket, n) is not getattr(_socket.socket, n)
        for n in network_guard.SOCKET_METHODS
    )
    guard.uninstall()
    _assert_pristine()


def test_the_guard_records_the_data_files_read(tmp_path: pathlib.Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    for name in ("a.csv.gz", "b.csv.gz"):
        (data / name).write_bytes(b"x")
    (tmp_path / "other.txt").write_bytes(b"x")
    guard = network_guard.install(data)
    try:
        with open(data / "a.csv.gz", "rb"):
            pass
        (data / "b.csv.gz").read_bytes()  # pathlib goes through io.open
        (tmp_path / "other.txt").read_bytes()  # outside data/: not a data read
        assert guard.reads == {"a.csv.gz", "b.csv.gz"}
        guard.check(["a.csv.gz", "b.csv.gz"])
        for expected in (["a.csv.gz"], ["a.csv.gz", "b.csv.gz", "c.csv.gz"], []):
            with pytest.raises(AssertionError, match=network_guard.READS_MESSAGE):
                guard.check(expected)
    finally:
        guard.uninstall()


@pytest.mark.parametrize("mode", ["w", "wb", "a", "ab", "x", "r+", "rb+", "w+b"])
def test_a_write_open_is_not_a_read(mode: str, tmp_path: pathlib.Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    target = data / ("new.csv.gz" if "x" in mode else "a.csv.gz")
    if "x" not in mode:
        target.write_bytes(b"x")
    guard = network_guard.install(data)
    try:
        with open(target, mode):
            pass
        with open(target, mode=mode.replace("x", "a")):  # keyword mode too
            pass
        assert guard.reads == set()
    finally:
        guard.uninstall()


def test_a_failed_open_is_not_a_read(tmp_path: pathlib.Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.csv.gz").write_bytes(b"x")
    guard = network_guard.install(data)
    try:
        with pytest.raises(FileExistsError), open(data / "a.csv.gz", "x"):
            pass
        with pytest.raises(FileNotFoundError), open(data / "missing.csv.gz", "rb"):
            pass
        assert guard.reads == set()
        with pytest.raises(AssertionError, match=network_guard.READS_MESSAGE):
            guard.check(["a.csv.gz"])
    finally:
        guard.uninstall()


def test_the_manifest_files_of_a_notebook() -> None:
    assert network_guard.manifest_files("tutorial_regression_tuning_lgbm.ipynb") == [
        "diamonds.csv.gz"
    ]
    assert network_guard.manifest_files("tutorial_calibration.ipynb") == []


FETCH = f"import urllib.request\nurllib.request.urlopen('https://{HOST}/x.csv')"
SWALLOWED = (
    "import urllib.request\n"
    "try:\n"
    f"    urllib.request.urlopen('https://{HOST}/x.csv')\n"
    "except Exception as exc:\n"
    "    assert 'H-0137' in str(exc)  # the call was made, and refused\n"
)
LOCAL_READ = (
    "import pandas as pd\n"
    f"pd.read_csv({str(index_runner.NOTEBOOKS / 'data' / 'vehicle.csv.gz')!r})"
)


def test_a_network_call_fails_the_instrumented_notebook() -> None:
    nb = _notebook(
        _cell(SETUP), _cell(FETCH), _cell("model.params_table()", tagged=True)
    )
    with pytest.raises(OSError, match=network_guard.MESSAGE):
        _exec(nb)


def test_a_swallowed_network_call_fails_at_the_last_cell() -> None:
    nb = _notebook(
        _cell(SETUP), _cell(SWALLOWED), _cell("model.params_table()", tagged=True)
    )
    with pytest.raises(AssertionError, match=network_guard.MESSAGE):
        _exec(nb)


def test_the_twin_without_a_network_call_passes() -> None:
    nb = _notebook(
        _cell(SETUP), _cell(LOCAL_READ), _cell("model.params_table()", tagged=True)
    )
    assert _exec(nb) == f"{index_runner.SENTINEL} 1"


VENDORED_READ = (
    "import pandas as pd\n"
    "assert pd.read_csv('data/credit-g.csv.gz').shape == (1000, 21)"
)
TAGGED = _cell("model.params_table()", tagged=True)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("case", "cells", "data_files", "reason"),
    [
        ("positive control", [TAGGED], (), None),
        (
            "a recorded data file read from the working directory",
            [_cell(VENDORED_READ), TAGGED],
            ("credit-g.csv.gz",),
            None,
        ),
        (
            "a data file the manifest does not record for the notebook",
            [_cell(VENDORED_READ), TAGGED],
            (),
            network_guard.READS_MESSAGE,
        ),
        (
            "a recorded data file the notebook does not read",
            [TAGGED],
            ("credit-g.csv.gz",),
            network_guard.READS_MESSAGE,
        ),
        ("a network call", [_cell(FETCH), TAGGED], (), network_guard.MESSAGE),
        (
            "a network call the notebook swallows",
            [_cell(SWALLOWED), TAGGED],
            (),
            network_guard.MESSAGE,
        ),
        (
            "a network call in a cell allowed to raise",
            [_cell(FETCH, tags=["raises-exception"]), TAGGED],
            (),
            network_guard.MESSAGE,
        ),
        (
            "a tagged cell the kernel skips",
            [_cell("model.params_table()", tagged=True, tags=["skip-execution"])],
            (),
            "never ran",
        ),
        (
            "a failed guard in a cell allowed to raise",
            [
                _cell(LOOK_ALIKE + "model = Fake()"),
                _cell("model.params_table()", tagged=True, tags=["raises-exception"]),
            ],
            (),
            "never ran",
        ),
        (
            "a receiver that is not a Model",
            [_cell(LOOK_ALIKE + "model = Fake()"), TAGGED],
            (),
            "not a Model",
        ),
    ],
)
def test_kernel_probes(
    case: str,
    cells: list[dict[str, Any]],
    data_files: tuple[str, ...],
    reason: str | None,
    tmp_path: pathlib.Path,
) -> None:
    del case
    root = pathlib.Path(__file__).resolve().parents[2]
    setup = _cell(f"import sys\nsys.path.insert(0, {str(root)!r})\n" + SETUP)
    nb = _notebook(setup, *cells)
    run = functools.partial(
        index_runner.run_instrumented,
        nb,
        workdir_base=tmp_path,
        timeout=120,
        data_files=data_files,
    )
    if reason is None:
        run()
    else:
        with pytest.raises(CellExecutionError, match=reason):
            run()
