"""Tests for engine.models.kalman."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.models import KalmanFilter, KalmanPairs


def test_kalman_constant_velocity_tracks_signal() -> None:
    # Constant-velocity 1-D tracker: state = [position, velocity].
    dt = 1.0
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * 1e-3
    R = np.array([[0.1]])
    kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x=np.array([0.0, 1.0]), P=np.eye(2))
    rng = np.random.default_rng(0)
    truth_positions = np.cumsum(np.ones(100))
    observations = truth_positions + rng.normal(0, 0.3, size=100)
    estimates = []
    for z in observations:
        kf.step(z)
        estimates.append(float(kf.x[0, 0]))
    estimates_arr = np.array(estimates)
    # After settling, tracking error should be smaller than raw observation error.
    track_err = np.abs(estimates_arr[20:] - truth_positions[20:]).mean()
    obs_err = np.abs(observations[20:] - truth_positions[20:]).mean()
    assert track_err < obs_err


def test_kalman_covariance_stays_symmetric_psd() -> None:
    F = np.eye(2)
    H = np.array([[1.0, 0.0]])
    kf = KalmanFilter(F=F, H=H, Q=np.eye(2) * 1e-4, R=np.array([[0.5]]), x=np.zeros(2), P=np.eye(2))
    for z in [1.0, 2.0, 1.5, 1.8, 2.2]:
        kf.step(z)
        # Symmetric
        np.testing.assert_allclose(kf.P, kf.P.T, atol=1e-10)
        # Positive semidefinite
        eigs = np.linalg.eigvalsh(kf.P)
        assert (eigs >= -1e-10).all()


def test_kalman_pairs_recovers_known_beta() -> None:
    rng = np.random.default_rng(0)
    n = 500
    true_beta = 1.5
    true_alpha = 0.2
    b = rng.normal(100.0, 5.0, size=n)
    a = true_alpha + true_beta * b + rng.normal(0, 0.5, size=n)
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    series_a = pd.Series(a, index=idx)
    series_b = pd.Series(b, index=idx)
    kp = KalmanPairs(delta=1e-5, R=0.25)
    df = kp.run(series_a, series_b)
    # After enough observations the recovered beta should be close to truth.
    assert abs(df["beta"].iloc[-1] - true_beta) < 0.1
    assert set(df.columns) == {"alpha", "beta", "spread", "sd", "z"}


def test_kalman_pairs_misaligned_index_raises() -> None:
    a = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2024-01-01", periods=3))
    b = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2024-02-01", periods=3))
    import pytest

    with pytest.raises(ValueError, match="share an index"):
        KalmanPairs().run(a, b)
