"""Two-state Gaussian Hidden Markov Model fit by EM.

A deliberately small, dependency-free implementation suitable for the
regime overlay in Module 16 chapter 1. Fits a 2-state HMM on a 1-D
observation series (typically log realised vol) and exposes the standard
forward-backward smoothing + Viterbi decoding APIs.

Why hand-rolled? hmmlearn is excellent but adds a heavy dependency for
one model. For the course's regime-overlay use case, this 150-line file
is enough — and importantly it stays callable on any environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _gauss_logpdf(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    """Log N(x; mu, var) for an array of x. Stable for small var."""
    var = max(var, 1e-12)
    return -0.5 * (np.log(2.0 * np.pi * var) + (x - mu) ** 2 / var)


def _logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Stable log-sum-exp."""
    m = np.max(a, axis=axis, keepdims=True)
    out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return np.squeeze(out, axis=axis) if axis is not None else out.reshape(())


@dataclass
class GaussianHMM2:
    """Two-state Gaussian HMM, EM-trained.

    State convention after :meth:`fit`: state 0 is the lower-mean state,
    state 1 is the higher-mean state. (Useful so "regime 0" always means
    the same thing across refits — e.g. low realised vol.)

    Attributes
    ----------
    means : (2,) per-state Gaussian means
    vars  : (2,) per-state Gaussian variances
    trans : (2, 2) row-stochastic transition matrix
    start : (2,) initial state distribution
    """

    means: np.ndarray = field(default_factory=lambda: np.zeros(2))
    vars: np.ndarray = field(default_factory=lambda: np.ones(2))
    trans: np.ndarray = field(default_factory=lambda: np.full((2, 2), 0.5))
    start: np.ndarray = field(default_factory=lambda: np.full(2, 0.5))
    n_iter_: int = 0
    log_likelihood_: float = 0.0

    # ------------------------------------------------------------------ EM

    def fit(
        self,
        x: np.ndarray,
        max_iter: int = 200,
        tol: float = 1e-6,
        seed: int = 0,
    ) -> GaussianHMM2:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size < 20:
            raise ValueError("need at least 20 observations to fit a 2-state HMM")
        rng = np.random.default_rng(seed)

        # Initialise: pick two quantile-based means + sample variance; mild
        # initial transition stickiness.
        self.means = np.array([np.quantile(x, 0.25), np.quantile(x, 0.75)])
        v = float(np.var(x))
        self.vars = np.array([v, v])
        self.trans = np.array([[0.95, 0.05], [0.05, 0.95]])
        self.start = np.array([0.5, 0.5])

        prev_ll = -np.inf
        for it in range(1, max_iter + 1):
            log_alpha, log_beta, ll = self._forward_backward(x)
            log_gamma = log_alpha + log_beta - ll  # posterior log-probs (T, 2)
            gamma = np.exp(log_gamma)

            # Pairwise posteriors (T-1, 2, 2)
            log_emit = self._log_emit(x)
            log_xi_numer = (
                log_alpha[:-1, :, None]
                + np.log(self.trans + 1e-300)[None, :, :]
                + log_emit[1:, None, :]
                + log_beta[1:, None, :]
            )
            log_xi = log_xi_numer - ll
            xi = np.exp(log_xi)

            # M-step.
            self.start = gamma[0] / gamma[0].sum()
            self.trans = xi.sum(axis=0) / np.clip(gamma[:-1].sum(axis=0)[:, None], 1e-12, None)
            for k in range(2):
                w = gamma[:, k]
                w_sum = w.sum()
                if w_sum < 1e-12:
                    # Avoid collapse by re-randomising this component.
                    self.means[k] = float(rng.choice(x))
                    self.vars[k] = v
                    continue
                self.means[k] = float((w * x).sum() / w_sum)
                self.vars[k] = float((w * (x - self.means[k]) ** 2).sum() / w_sum) + 1e-8

            if abs(ll - prev_ll) < tol:
                self.n_iter_ = it
                self.log_likelihood_ = float(ll)
                break
            prev_ll = ll
        else:
            self.n_iter_ = max_iter
            self.log_likelihood_ = float(prev_ll)

        # Canonicalise: state 0 = lower mean.
        if self.means[0] > self.means[1]:
            self.means = self.means[::-1].copy()
            self.vars = self.vars[::-1].copy()
            self.trans = self.trans[::-1, ::-1].copy()
            self.start = self.start[::-1].copy()

        return self

    # --------------------------------------------------------- inference

    def _log_emit(self, x: np.ndarray) -> np.ndarray:
        """(T, 2) log emission probabilities."""
        return np.column_stack(
            [
                _gauss_logpdf(x, self.means[0], self.vars[0]),
                _gauss_logpdf(x, self.means[1], self.vars[1]),
            ]
        )

    def _forward_backward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        T = x.shape[0]
        log_emit = self._log_emit(x)
        log_trans = np.log(self.trans + 1e-300)
        log_start = np.log(self.start + 1e-300)

        # Forward
        log_alpha = np.full((T, 2), -np.inf)
        log_alpha[0] = log_start + log_emit[0]
        for t in range(1, T):
            for k in range(2):
                log_alpha[t, k] = _logsumexp(log_alpha[t - 1] + log_trans[:, k]) + log_emit[t, k]
        ll = float(_logsumexp(log_alpha[-1]))

        # Backward
        log_beta = np.full((T, 2), -np.inf)
        log_beta[-1] = 0.0
        for t in range(T - 2, -1, -1):
            for k in range(2):
                log_beta[t, k] = _logsumexp(log_trans[k] + log_emit[t + 1] + log_beta[t + 1])
        return log_alpha, log_beta, ll

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Smoothed posterior P(state | observations) — shape (T, 2)."""
        x = np.asarray(x, dtype=float).reshape(-1)
        log_alpha, log_beta, ll = self._forward_backward(x)
        return np.exp(log_alpha + log_beta - ll)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Viterbi most-likely state sequence — shape (T,)."""
        x = np.asarray(x, dtype=float).reshape(-1)
        T = x.shape[0]
        log_emit = self._log_emit(x)
        log_trans = np.log(self.trans + 1e-300)
        log_start = np.log(self.start + 1e-300)

        delta = np.full((T, 2), -np.inf)
        psi = np.zeros((T, 2), dtype=int)
        delta[0] = log_start + log_emit[0]
        for t in range(1, T):
            for k in range(2):
                cand = delta[t - 1] + log_trans[:, k]
                psi[t, k] = int(np.argmax(cand))
                delta[t, k] = cand[psi[t, k]] + log_emit[t, k]

        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path
