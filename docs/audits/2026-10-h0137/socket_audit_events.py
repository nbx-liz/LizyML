"""Which audit events does each socket operation raise? (H-0137 revision 3)

Calls every public ``socket`` function and ``socket.socket`` method that takes
a host, a service, a protocol name or an address, under an audit hook that
records the ``socket.*`` events, and prints one line per call. Hosts are
``.invalid`` names, so nothing resolves; failures are expected and printed.
The ``_socket`` route and ``create_server`` / ``bind`` with a host name (the
round-3 finding) are included.

    uv run python docs/audits/2026-10-h0137/socket_audit_events.py
"""

from __future__ import annotations

import _socket
import socket
import sys
from collections.abc import Callable

HOST = "lizyml-h0137.invalid"
LOOPBACK = ("127.0.0.1", 9)
events: list[str] = []


def hook(event: str, args: tuple[object, ...]) -> None:
    if event.startswith("socket."):
        events.append(event)


def udp() -> socket.socket:
    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


CALLS: dict[str, Callable[[], object]] = {
    "socket.socket.bind(host)": lambda: udp().bind((HOST, 0)),
    "socket.socket.bind(loopback)": lambda: udp().bind(("127.0.0.1", 0)),
    "socket.socket.connect": lambda: socket.socket().connect((HOST, 443)),
    "socket.socket.connect_ex": lambda: socket.socket().connect_ex((HOST, 443)),
    "socket.socket.sendto": lambda: udp().sendto(b"x", (HOST, 9)),
    "socket.socket.sendmsg": lambda: udp().sendmsg([b"x"], [], 0, (HOST, 9)),
    "socket.create_connection": lambda: socket.create_connection((HOST, 443), 1),
    "socket.create_server": lambda: socket.create_server((HOST, 0)),
    "socket.getaddrinfo": lambda: socket.getaddrinfo(HOST, 443),
    "socket.getfqdn": lambda: socket.getfqdn(HOST),
    "socket.gethostbyaddr": lambda: socket.gethostbyaddr("127.0.0.1"),
    "socket.gethostbyname": lambda: socket.gethostbyname(HOST),
    "socket.gethostbyname_ex": lambda: socket.gethostbyname_ex(HOST),
    "socket.getnameinfo": lambda: socket.getnameinfo(LOOPBACK, 0),
    "socket.getprotobyname": lambda: socket.getprotobyname("tcp"),
    "socket.getservbyname": lambda: socket.getservbyname("https"),
    "socket.getservbyport": lambda: socket.getservbyport(443),
    "socket.send_fds": lambda: socket.send_fds(udp(), [b"x"], [], 0, (HOST, 9)),
    "socket.gethostname": socket.gethostname,
    "socket.inet_aton": lambda: socket.inet_aton("127.0.0.1"),
    "_socket.getaddrinfo": lambda: _socket.getaddrinfo(HOST, 443),
    "_socket.socket.connect": lambda: _socket.socket().connect((HOST, 443)),
}

sys.addaudithook(hook)
print(sys.version.split()[0])
for name, call in CALLS.items():
    events.clear()
    try:
        call()
        outcome = "ok"
    except Exception as exc:  # the probe reports every failure instead of stopping
        outcome = type(exc).__name__
    print(f"{name:32} {outcome:12} {sorted(set(events))}")
