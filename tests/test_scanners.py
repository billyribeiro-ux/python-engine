"""Tests for the scanner registry and protocol."""

from __future__ import annotations

import pandas as pd
import pytest

from engine.scanners import Scanner, registry, scanner


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    """Each test gets a fresh registry slate."""
    monkeypatch.setattr(registry, "_data", {})


def test_scanner_decorator_registers_class() -> None:
    @scanner("test-scanner")
    class MyScanner:
        def scan(self, universe, asof):
            return pd.DataFrame({"symbol": list(universe)})

    assert "test-scanner" in registry
    assert registry.get("test-scanner") is MyScanner
    assert MyScanner.name == "test-scanner"


def test_duplicate_name_raises() -> None:
    @scanner("dupe")
    class A:
        def scan(self, universe, asof):
            return pd.DataFrame()

    with pytest.raises(ValueError, match="duplicate scanner"):

        @scanner("dupe")
        class B:
            def scan(self, universe, asof):
                return pd.DataFrame()


def test_unknown_name_raises() -> None:
    with pytest.raises(KeyError):
        registry.get("does-not-exist")


def test_class_must_implement_scan() -> None:
    with pytest.raises(TypeError, match="must implement scan"):

        @scanner("bad")
        class Bad:
            pass


def test_protocol_satisfaction() -> None:
    @scanner("ok")
    class GoodScanner:
        def scan(self, universe, asof):
            return pd.DataFrame()

    s = GoodScanner()
    assert isinstance(s, Scanner)
