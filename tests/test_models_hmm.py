"""Tests for engine.models.hmm."""

from __future__ import annotations

import numpy as np
import pytest

from engine.models import GaussianHMM2


def _synthetic_two_regime(n: int = 500, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate observations from a known 2-state Markov chain.

    Returns (observations, true_states).
    """
    rng = np.random.default_rng(seed)
    trans = np.array([[0.97, 0.03], [0.05, 0.95]])
    means = np.array([0.0, 3.0])
    sds = np.array([0.5, 0.5])
    states = np.zeros(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        states[t] = int(rng.choice(2, p=trans[states[t - 1]]))
    obs = rng.normal(means[states], sds[states])
    return obs, states


def test_hmm_recovers_two_means() -> None:
    obs, _ = _synthetic_two_regime(n=800, seed=1)
    hmm = GaussianHMM2().fit(obs)
    # State 0 is canonicalised to lower mean.
    assert hmm.means[0] < hmm.means[1]
    assert abs(hmm.means[0] - 0.0) < 0.3
    assert abs(hmm.means[1] - 3.0) < 0.3


def test_hmm_viterbi_majority_correct() -> None:
    obs, truth = _synthetic_two_regime(n=800, seed=2)
    hmm = GaussianHMM2().fit(obs)
    pred = hmm.predict(obs)
    # Allow for label swap: take the better of (pred, 1-pred).
    agree = max((pred == truth).mean(), ((1 - pred) == truth).mean())
    assert agree > 0.85


def test_hmm_predict_proba_sums_to_one() -> None:
    obs, _ = _synthetic_two_regime(n=200, seed=3)
    hmm = GaussianHMM2().fit(obs)
    proba = hmm.predict_proba(obs)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(obs)), atol=1e-6)


def test_hmm_rejects_too_short_input() -> None:
    with pytest.raises(ValueError, match="at least 20"):
        GaussianHMM2().fit(np.array([1.0, 2.0, 3.0]))
