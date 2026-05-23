"""Cross-cutting utilities used by examples in the course."""

from __future__ import annotations

import socket
from contextlib import contextmanager
from time import perf_counter


@contextmanager
def timed(label: str = "block"):
    """Tiny ``with`` block that prints how long the body took.

    >>> with timed("sleep"):
    ...     import time; time.sleep(0.01)  # doctest: +SKIP
    sleep: 10.3 ms
    """
    t0 = perf_counter()
    try:
        yield
    finally:
        dt = (perf_counter() - t0) * 1000
        print(f"{label}: {dt:.1f} ms")


def has_network(host: str = "1.1.1.1", port: int = 53, timeout: float = 1.0) -> bool:
    """Cheap reachability check used to skip vendor tests when offline.

    We hit Cloudflare's DNS rather than the vendor so a failing test means
    the network is gone, not that the vendor is rate-limiting us.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


__all__ = ["has_network", "timed"]
