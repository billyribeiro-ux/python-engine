"""Kalman filter and a pairs-trading specialisation.

The general :class:`KalmanFilter` is a 16-line linear-Gaussian state
estimator with optional control input. :class:`KalmanPairs` is the
pairs-trading specialisation from Module 8 chapter 4: the state is
``[alpha, beta]``, the observation is asset A, and the design matrix at
each step is ``[1, B_t]`` so the slope evolves as a random walk.

Both store a covariance ``P`` and update it via the standard (Joseph-form)
equation for numerical stability.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
import pandas as pd

ArrayLike: TypeAlias = "np.ndarray | list | tuple"


def _atleast_2d(x: ArrayLike) -> np.ndarray:
    arr = np.atleast_2d(np.asarray(x, dtype=float))
    return arr


@dataclass
class KalmanFilter:
    """Linear-Gaussian Kalman filter.

    State equation:  ``x_t = F x_{t-1} + B u_t + w_t``,  ``w ~ N(0, Q)``
    Observation:    ``z_t = H x_t + v_t``,                ``v ~ N(0, R)``

    Parameters
    ----------
    F : (n, n) state-transition matrix
    H : (m, n) observation matrix
    Q : (n, n) process noise covariance
    R : (m, m) observation noise covariance
    x0 : (n,) initial mean
    P0 : (n, n) initial covariance
    B : (n, k) optional control matrix (None = no control input)
    """

    F: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    x: np.ndarray
    P: np.ndarray
    B: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.F = np.asarray(self.F, dtype=float)
        self.H = np.asarray(self.H, dtype=float)
        self.Q = np.asarray(self.Q, dtype=float)
        self.R = np.asarray(self.R, dtype=float)
        self.x = np.asarray(self.x, dtype=float).reshape(-1, 1)
        self.P = np.asarray(self.P, dtype=float)
        if self.B is not None:
            self.B = np.asarray(self.B, dtype=float)

    def predict(self, u: ArrayLike | None = None) -> None:
        """Advance the state one step (no observation)."""
        if self.B is not None and u is not None:
            u_arr = np.asarray(u, dtype=float).reshape(-1, 1)
            self.x = self.F @ self.x + self.B @ u_arr
        else:
            self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: ArrayLike) -> tuple[float, float]:
        """Incorporate an observation. Returns (innovation, innovation_std)."""
        z_arr = np.asarray(z, dtype=float).reshape(-1, 1)
        y = z_arr - self.H @ self.x  # innovation
        S = self.H @ self.P @ self.H.T + self.R  # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        # Joseph form: (I - KH) P (I - KH)^T + K R K^T — symmetric, PSD-preserving.
        I_KH = np.eye(self.x.shape[0]) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return float(y[0, 0]), float(np.sqrt(S[0, 0]))

    def step(self, z: ArrayLike, u: ArrayLike | None = None) -> tuple[float, float]:
        """Convenience: predict, then update."""
        self.predict(u)
        return self.update(z)


@dataclass
class KalmanPairs:
    """Online estimator of the time-varying hedge ratio for a pair.

    Models ``A_t = alpha_t + beta_t * B_t + v_t`` with the parameter pair
    evolving as a random walk. Recovers ``alpha_t, beta_t`` plus the
    standardised spread ``z_t`` — the trading signal — bar by bar.

    Parameters
    ----------
    delta : controls the random-walk variance on (alpha, beta).
            Q = delta / (1 - delta) * I.  Smaller = more stable; larger
            = faster adaptation. Typical range 1e-5 .. 1e-3.
    R : observation noise variance on A.
    P0 : initial parameter uncertainty (default 1.0).
    """

    delta: float = 1e-4
    R: float = 1.0
    P0: float = 1.0
    x: np.ndarray = field(init=False)
    P: np.ndarray = field(init=False)
    Q: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.x = np.zeros((2, 1))
        self.P = np.eye(2) * self.P0
        self.Q = np.eye(2) * (self.delta / (1.0 - self.delta))

    @property
    def alpha(self) -> float:
        return float(self.x[0, 0])

    @property
    def beta(self) -> float:
        return float(self.x[1, 0])

    def update(self, a: float, b: float) -> tuple[float, float]:
        """One step.

        Returns (spread, spread_std) — the innovation and its 1-sigma scale.
        The trading signal is z = spread / spread_std.
        """
        H = np.array([[1.0, float(b)]])
        z = np.array([[float(a)]])
        # Predict: F = I, no control input → only inflate covariance.
        self.P = self.P + self.Q
        # Update.
        y = z - H @ self.x
        S = H @ self.P @ H.T + np.array([[self.R]])
        K = self.P @ H.T / S
        self.x = self.x + K @ y
        I_KH = np.eye(2) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ np.array([[self.R]]) @ K.T
        spread = float(y[0, 0])
        sd = float(np.sqrt(S[0, 0]))
        return spread, sd

    def run(self, a: pd.Series, b: pd.Series) -> pd.DataFrame:
        """Vectorised driver over aligned series.

        Returns a DataFrame indexed like the inputs with columns
        ``alpha, beta, spread, sd, z``.
        """
        if not a.index.equals(b.index):
            raise ValueError("a and b must share an index")
        rows: list[tuple[float, float, float, float, float]] = []
        for av, bv in zip(a.values, b.values, strict=True):
            spread, sd = self.update(float(av), float(bv))
            z = spread / sd if sd > 0 else 0.0
            rows.append((self.alpha, self.beta, spread, sd, z))
        return pd.DataFrame(rows, index=a.index, columns=["alpha", "beta", "spread", "sd", "z"])

    def __iter__(self) -> Iterator[tuple[float, float]]:
        # Pure type-hint convenience; not commonly used.
        return iter([])
