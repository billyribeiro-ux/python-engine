"""Regenerate the chart assets used in the course.

Run with ``make assets``. The script is idempotent and overwrites images in
``course/_assets/``. Charts here are intentionally small and self-contained
so the course renders identically with or without internet access.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    print("matplotlib not installed; install with `pip install -e .[core]`")
    raise SystemExit(1) from exc


ASSETS = Path(__file__).resolve().parent.parent / "course" / "_assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def fig_broadcasting() -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    rows = np.arange(4)[:, None]
    cols = np.arange(5)[None, :]
    grid = rows * 10 + cols
    ax.imshow(grid, cmap="Blues")
    for i in range(4):
        for j in range(5):
            ax.text(j, i, int(grid[i, j]), ha="center", va="center", color="black")
    ax.set_title("Broadcasting: (4,1) + (1,5) → (4,5)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(ASSETS / "broadcasting.png", dpi=140)
    plt.close(fig)


def fig_strides() -> None:
    fig, ax = plt.subplots(figsize=(7, 2.4))
    arr = np.arange(8)
    ax.imshow(arr.reshape(1, -1), cmap="Greens", aspect="auto")
    for i, v in enumerate(arr):
        ax.text(i, 0, str(v), ha="center", va="center", color="black")
    ax.set_yticks([])
    ax.set_xticks(range(8))
    ax.set_title("Contiguous int64 array: stride = 8 bytes per step")
    fig.tight_layout()
    fig.savefig(ASSETS / "strides.png", dpi=140)
    plt.close(fig)


def fig_rolling_window() -> None:
    arr = np.arange(10)
    win = 3
    rows = arr.size - win + 1
    fig, ax = plt.subplots(figsize=(7, 3))
    grid = np.array([arr[i : i + win] for i in range(rows)])
    ax.imshow(grid, cmap="Purples")
    for i in range(rows):
        for j in range(win):
            ax.text(j, i, int(grid[i, j]), ha="center", va="center", color="black")
    ax.set_title("sliding_window_view: 10 → (8, 3)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(ASSETS / "rolling_window.png", dpi=140)
    plt.close(fig)


def main() -> None:
    fig_broadcasting()
    fig_strides()
    fig_rolling_window()
    print(f"Wrote charts to {ASSETS}")


if __name__ == "__main__":
    main()
