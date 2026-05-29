"""Time-series cross-validation that doesn't leak.

Implements purged k-fold and walk-forward splits with an explicit embargo
period, after López de Prado (2018). These splitters return ``(train, test)``
index pairs you can plug into any scikit-learn-style training loop.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PurgedKFold:
    """k-fold over a time-ordered index with a configurable purge/embargo.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    purge : int
        Number of bars to drop from the *end* of each training fold to
        prevent the label horizon of train samples leaking into the test
        window. Set to your label horizon (e.g. ``5`` for a 5-day forward
        return label).
    embargo : int
        Number of bars to drop from the *start* of training following each
        test window, for the same purpose in the other direction.
    """

    n_splits: int = 5
    purge: int = 0
    embargo: int = 0

    def split(self, n: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        fold_sizes = np.full(self.n_splits, n // self.n_splits)
        fold_sizes[: n % self.n_splits] += 1
        bounds: list[tuple[int, int]] = []
        start = 0
        for size in fold_sizes:
            bounds.append((start, start + size))
            start += size

        all_idx = np.arange(n)
        for _k, (a, b) in enumerate(bounds):
            test = all_idx[a:b]
            keep = np.ones(n, dtype=bool)
            keep[a:b] = False
            # purge: drop the `purge` bars before the test fold
            if self.purge > 0:
                left = max(0, a - self.purge)
                keep[left:a] = False
            # embargo: drop the `embargo` bars after the test fold
            if self.embargo > 0:
                right = min(n, b + self.embargo)
                keep[b:right] = False
            train = all_idx[keep]
            yield train, test


@dataclass(frozen=True, slots=True)
class WalkForward:
    """Expanding-window walk-forward CV.

    On step k, train on ``[0, train_end_k]`` and test on
    ``[train_end_k + embargo, train_end_k + embargo + test_size]``.

    Parameters
    ----------
    initial_train : int
        Size of the very first training window.
    test_size : int
        Bars per test fold.
    embargo : int
        Bars between train end and test start.
    """

    initial_train: int
    test_size: int
    embargo: int = 0

    def split(self, n: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        all_idx = np.arange(n)
        cur = self.initial_train
        while cur + self.embargo + self.test_size <= n:
            train = all_idx[:cur]
            test_start = cur + self.embargo
            test = all_idx[test_start : test_start + self.test_size]
            yield train, test
            cur += self.test_size


def deflated_sharpe(observed: float, sharpes: Iterable[float], T: int) -> float:
    """Probability that the maximum observed Sharpe across N trials is real.

    Implements the approximation from Bailey & Lopez de Prado (2014). Pass
    the full list of trial Sharpes (the implicit selection set you searched
    over) and the in-sample length T. Returns a probability in [0, 1] — close
    to 1 means "very likely real", close to 0 means "looks like selection".
    """
    from scipy.stats import norm

    arr = np.asarray(list(sharpes), dtype=float)
    n_trials = len(arr)
    if n_trials == 0:
        return float("nan")
    sd = float(arr.std(ddof=1)) if n_trials > 1 else 0.0
    e_max = sd * (
        (1 - np.euler_gamma) * norm.ppf(1 - 1 / n_trials)
        + np.euler_gamma * norm.ppf(1 - 1 / (n_trials * np.e))
    )
    # standard error of the observed sharpe, under the null
    se = np.sqrt((1 - float(arr.mean()) * observed + 0.5 * observed**2) / max(T - 1, 1))
    if se <= 0:
        return float("nan")
    return float(norm.cdf((observed - e_max) / se))
