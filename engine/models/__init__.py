"""Models: Kalman filters and HMM regime detection.

Production-grade implementations of the algorithms walked through in
Modules 7, 8, and 16 of the course. Pure NumPy / SciPy — no heavy
dependencies — so they install cleanly and run anywhere.
"""

from __future__ import annotations

from .hmm import GaussianHMM2
from .kalman import KalmanFilter, KalmanPairs

__all__ = ["GaussianHMM2", "KalmanFilter", "KalmanPairs"]
