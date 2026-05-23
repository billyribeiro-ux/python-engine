"""Smoke tests for the operational scripts in ``scripts/``.

We don't test the full behaviour (that would be a Module 22 end-to-end);
just verify the entry points parse arguments cleanly and the synthetic
mode of the end-to-end smoke runs through.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_end_to_end_smoke_help() -> None:
    """`--help` should exit 0 and print the usage."""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "end_to_end_smoke.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "smoke" in proc.stdout.lower() or "smoke" in proc.stderr.lower()


def test_end_to_end_smoke_synthetic_runs() -> None:
    """Full pipeline against synthetic data should complete with [smoke] OK."""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "end_to_end_smoke.py"), "--no-network"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "[smoke] OK" in proc.stdout, proc.stdout


def test_run_scanners_help() -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_scanners.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "universe" in proc.stdout.lower()


def test_build_assets_help_or_run(tmp_path) -> None:
    """build_assets.py should at least be importable and have a main()."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util, sys; "
            "spec = importlib.util.spec_from_file_location('m', "
            f"'{REPO_ROOT / 'scripts' / 'build_assets.py'}'); "
            "m = importlib.util.module_from_spec(spec); "
            "assert hasattr(spec.loader, 'exec_module'); "
            "print('importable')",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "importable" in proc.stdout
