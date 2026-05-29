"""Scanner framework.

A scanner is anything that takes a universe of symbols (and an as-of time)
and returns a DataFrame of *candidates* — symbols with their per-symbol
metric values, sorted by score.

The framework's value is in the registry: write a scanner, decorate it with
``@scanner("my-name")``, and it auto-registers. The runtime can then list
every scanner, instantiate by name, and run uniformly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Scanner(Protocol):
    """Anything with ``name`` and a ``scan`` method satisfying this shape."""

    name: str

    def scan(self, universe: Sequence[str], asof: pd.Timestamp) -> pd.DataFrame: ...


class _Registry:
    """Singleton-style registry. Use the module-level ``registry`` instance."""

    def __init__(self) -> None:
        self._data: dict[str, type] = {}

    def register(self, name: str, cls: type) -> type:
        if name in self._data:
            raise ValueError(f"duplicate scanner: {name}")
        self._data[name] = cls
        return cls

    def get(self, name: str) -> type:
        if name not in self._data:
            raise KeyError(f"no scanner registered as {name!r}; have {sorted(self._data)}")
        return self._data[name]

    def names(self) -> list[str]:
        return sorted(self._data)

    def __contains__(self, name: str) -> bool:
        return name in self._data


registry = _Registry()


def scanner(name: str) -> Callable[[type], type]:
    """Class decorator that registers the scanner under ``name``.

    The decorated class must implement ``scan(self, universe, asof)``. The
    decorator sets ``cls.name`` to the registry name and adds it to the
    global registry.
    """

    def deco(cls: type) -> type:
        if not hasattr(cls, "scan"):
            raise TypeError(f"{cls!r} must implement scan(universe, asof)")
        setattr(cls, "name", name)  # noqa: B010 - dynamic attribute on the class
        return registry.register(name, cls)

    return deco
