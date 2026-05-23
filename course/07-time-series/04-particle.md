# Particle filters

The Kalman filter requires linearity and Gaussian noise. The world doesn't oblige. Particle filters — also called Sequential Monte Carlo — give you the same online state-estimation machinery for *any* dynamical system, at the cost of running thousands of Monte Carlo samples instead of two matrix equations.

For trading, they show up in: stochastic-volatility model estimation, non-Gaussian regime tracking, online inference for systems with non-linear observation equations (options pricing as a state-function).

## The setup

State equation (now arbitrary):
$$
x_t = f(x_{t-1}, w_t)
$$

Observation equation (also arbitrary):
$$
z_t = h(x_t, v_t)
$$

The posterior $p(x_t | z_{1:t})$ has no closed-form. The particle filter approximates it with $N$ weighted samples ("particles"):

$$
p(x_t | z_{1:t}) \approx \sum_{i=1}^N w_t^{(i)} \delta(x_t - x_t^{(i)})
$$

## The algorithm

For each step $t$:

1. **Propagate** each particle through the state equation, sampling new states.
2. **Weight** each new particle by the observation likelihood $p(z_t | x_t^{(i)})$.
3. **Normalise** weights to sum to 1.
4. **Resample** with replacement according to the weights (when effective sample size drops too low).

```python
import numpy as np
rng = np.random.default_rng(0)

class ParticleFilter:
    def __init__(self, n_particles, state_transition, obs_log_likelihood, init_state):
        self.N = n_particles
        self.f = state_transition           # x_t = f(x_{t-1}) — returns array of shape (N,)
        self.log_lik = obs_log_likelihood   # log p(z | x) for an array of x's
        self.x = np.asarray(init_state)
        self.w = np.ones(n_particles) / n_particles

    def step(self, z):
        # 1. propagate
        self.x = self.f(self.x)
        # 2. weight in log space (avoid underflow)
        log_w = np.log(self.w + 1e-300) + self.log_lik(self.x, z)
        # 3. normalise via log-sum-exp
        m = log_w.max()
        w = np.exp(log_w - m); w /= w.sum()
        self.w = w
        # 4. resample if ESS too low
        ess = 1.0 / (w ** 2).sum()
        if ess < self.N / 2:
            idx = rng.choice(self.N, size=self.N, p=w, replace=True)
            self.x = self.x[idx]
            self.w[:] = 1.0 / self.N

    def estimate(self):
        return (self.w * self.x).sum()
```

That's the bootstrap particle filter, the simplest variant. Sixty lines including a numerically-stable weight update.

## A worked example: stochastic volatility

Model:
$$
\log \sigma_t^2 = \alpha + \beta \log \sigma_{t-1}^2 + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \sigma_\eta^2)
$$
$$
r_t = \sigma_t z_t, \quad z_t \sim \mathcal{N}(0, 1)
$$

The hidden state is $\log \sigma_t^2$. We observe $r_t$. The Kalman filter doesn't apply (the observation is non-linear in the state). The particle filter does.

```python
alpha, beta, sigma_eta = -0.02, 0.95, 0.25

def transition(x):
    return alpha + beta * x + rng.normal(0, sigma_eta, size=x.shape)

def log_lik(x, r):
    """log N(r; 0, exp(x))"""
    return -0.5 * (np.log(2 * np.pi) + x + r ** 2 / np.exp(x))

pf = ParticleFilter(
    n_particles=5_000,
    state_transition=transition,
    obs_log_likelihood=log_lik,
    init_state=rng.normal(np.log(0.01 ** 2), 0.5, size=5_000),
)

est_log_var = []
for r in returns:
    pf.step(r)
    est_log_var.append(pf.estimate())

import numpy as np
sigma_path = np.exp(np.array(est_log_var) / 2)         # back to vol
```

`sigma_path` is the filter's online estimate of the latent volatility. Plot it next to a GARCH or EWMA estimate; they'll be similar in calm regimes and diverge during stress, where the SV model's separate-noise-on-vol gives a more honest uncertainty.

## A worked example: regime detection with non-Gaussian likelihoods

Hidden state: regime ∈ {1, 2}. Transition: small probability of switching each day. Observation: return ~ Student-t with regime-specific scale and df.

```python
# State propagation: with prob 0.02 switch, otherwise stay
def transition_regime(x):
    switch = rng.random(size=x.shape) < 0.02
    return np.where(switch, 3 - x, x)         # toggles 1 ↔ 2

# Observation: t-distribution with regime-specific (sigma, df)
from scipy.stats import t
sigmas = {1: 0.005, 2: 0.02}
dfs = {1: 8.0, 2: 4.0}

def log_lik_regime(x, r):
    s = np.array([sigmas[int(round(xi))] for xi in x])
    d = np.array([dfs[int(round(xi))] for xi in x])
    return t.logpdf(r / s, df=d) - np.log(s)
```

The particle filter handles all of it because the algorithm doesn't care about distributional shapes — only that you can sample from the state transition and evaluate the observation log-likelihood.

## Variants worth knowing

- **Bootstrap PF** — what we built above. Simple, robust, works.
- **Sequential Importance Resampling (SIR)** — same thing, different name.
- **Auxiliary PF** — biases the propagation toward states with high observation likelihood; better in narrow-likelihood scenarios.
- **Rao-Blackwellised PF** — when part of the state is Kalman-tractable, sample only the non-Gaussian part; the Gaussian part is handled exactly.

For production work on real models, the standard library is [`particles`](https://particles-sequential-monte-carlo-in-python.readthedocs.io/) (by Nicolas Chopin). It implements all of the above plus the sophisticated samplers (PMMH for parameter estimation).

## Particle Markov Chain Monte Carlo (PMMH)

A particle filter gives you state estimates given fixed parameters. PMMH wraps it in MCMC so you can also estimate the parameters:

```python
# Pseudocode
for iteration in range(N_MCMC):
    theta_prop = propose(theta_current)
    # Use particle filter to compute the *marginal likelihood* p(data | theta)
    marginal_prop = particle_filter(data, theta_prop).marginal_likelihood
    marginal_curr = particle_filter(data, theta_current).marginal_likelihood
    if accept(marginal_prop, marginal_curr):
        theta_current = theta_prop
```

PMMH is computationally heavy (it runs a particle filter per MCMC step), but it's the right tool when you have a model where the parameters are unknown and the state is non-Gaussian. For stochastic vol estimation, it gives you full posterior distributions over $\alpha, \beta, \sigma_\eta$ in addition to the latent vol path.

## Pitfalls

!!! warning "Sample impoverishment"
    After many resampling steps, particles collapse to a few unique values. The fix: add a small jitter to particles after resampling, or use the auxiliary PF, or simply use more particles.

!!! warning "Particle filter for parameter estimation directly is wrong"
    A particle filter estimates *states*, not parameters. Treating parameters as states with zero process noise will give degenerate filters. Use PMMH or grid-search parameters offline.

!!! warning "N=100 isn't enough for serious work"
    For a multi-dimensional state, you typically need $N \geq 1000$ particles for stable estimates. For PMMH, $N \geq 10{,}000$ is normal. Particle filters are not free.

!!! warning "Compute scaling"
    Pure-Python PFs are slow. Vectorise the propagation and likelihood with NumPy (as in the example above) or JIT with numba. For real-time use, the inner step needs to be sub-millisecond.

## Bottom line

Reach for particle filters when:

- The state-space model is non-linear or non-Gaussian.
- You're modelling **stochastic volatility** properly (not just GARCH-approximated).
- You're tracking regime / state in a Bayesian way and the Kalman approximation isn't good enough.

For Kalman-tractable problems, use Kalman. For toy MCMC of static parameters, use PyMC. Particle filters are the heavy machinery for the genuinely difficult middle.

Continue to **[Wavelets and the Hilbert transform](05-wavelets.md)**.
