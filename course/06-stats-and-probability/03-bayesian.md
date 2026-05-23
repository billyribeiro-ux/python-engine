# Bayesian thinking for traders

In a low-signal world, the prior dominates. Bayesian inference is the formal apparatus for combining what you already believe with what you just measured, in a way that handles uncertainty honestly. For trading, the biggest practical wins are: shrinkage of strategy means, sequential updating during live trading, and explicit prior-posterior reasoning about regime change.

This chapter is opinionated and brief. It teaches you the *shape* of Bayesian thinking and points at PyMC for the heavy lifting. Don't memorise Bayes' rule; internalise the picture.

## The picture

Three distributions matter:

- **Prior** $p(\theta)$ — what you believe about $\theta$ before seeing the data.
- **Likelihood** $p(D \mid \theta)$ — how plausible the data is under each value of $\theta$.
- **Posterior** $p(\theta \mid D) \propto p(D \mid \theta) \cdot p(\theta)$ — what you believe after seeing the data.

The posterior is just the product of prior and likelihood, normalised. Three lines of algebra; an entire research methodology built on top.

## Conjugate examples — when the math is closed-form

### Mean of a normal with known variance

Prior: $\mu \sim \mathcal{N}(\mu_0, \tau_0^2)$. Data: $n$ samples with mean $\bar x$ and known variance $\sigma^2$.

Posterior mean is a precision-weighted average:

$$
\mu_{\text{post}} = \frac{\tau_0^{-2} \mu_0 + n\sigma^{-2} \bar x}{\tau_0^{-2} + n\sigma^{-2}}
$$

This *is* shrinkage. When $n$ is small, the posterior leans on the prior. When $n$ is large, the data wins.

```python
import numpy as np

def posterior_mean_normal(prior_mean, prior_sd, data_mean, data_sd, n):
    prec0 = 1 / prior_sd**2
    prec_data = n / data_sd**2
    return (prec0 * prior_mean + prec_data * data_mean) / (prec0 + prec_data)


# Strategy with apparent Sharpe 1.5, 2 years of data, but prior is "true Sharpe ~0"
posterior_sharpe = posterior_mean_normal(
    prior_mean=0.0, prior_sd=0.3,      # prior: most strategies have low edge
    data_mean=1.5, data_sd=1/np.sqrt(252*2),
    n=1,
)
print(f"Posterior Sharpe: {posterior_sharpe:.2f}")
```

The prior pulls the noisy estimate substantially toward zero — which is honest, given how easy it is to luck into a backtest Sharpe of 1.5.

### Beta-binomial — probability of win

A natural pairing for "what's the probability my next trade wins?"

Prior: $p \sim \text{Beta}(\alpha_0, \beta_0)$.
Data: $w$ wins, $\ell$ losses.
Posterior: $p \sim \text{Beta}(\alpha_0 + w, \beta_0 + \ell)$.

```python
from scipy.stats import beta

# Prior: I think win rate is around 50%, with moderate uncertainty
a, b = 5, 5
# After observing 30 wins and 20 losses
a_post, b_post = a + 30, b + 20

# 95% credible interval for the true win rate
ci = beta.interval(0.95, a_post, b_post)
print(f"True win-rate 95% CI: {ci}")
print(f"Posterior mean win-rate: {beta.mean(a_post, b_post):.2%}")
```

That's a complete, principled answer with no MCMC required. The Beta-Binomial pairing is the right tool for "should I keep using this strategy?" decisions made on small samples.

## When closed-form isn't enough — PyMC

For models with non-conjugate priors, multiple parameters, or hierarchies, you need MCMC (Markov chain Monte Carlo). PyMC is the standard:

```python
import pymc as pm
import numpy as np

returns = your_strategy_returns        # shape (T,)

with pm.Model() as m:
    # Priors
    mu = pm.Normal("mu", 0.0, 0.0005)          # daily mean ~ 0
    sigma = pm.HalfNormal("sigma", 0.02)
    nu = pm.Exponential("nu", 1/10) + 2        # degrees of freedom > 2

    # Likelihood — Student-t, because returns have fat tails
    pm.StudentT("obs", mu=mu, sigma=sigma, nu=nu, observed=returns)

    trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.95)

print(pm.summary(trace, var_names=["mu", "sigma", "nu"]))
```

What you get back: posterior samples of each parameter. You can compute any quantity from them — the posterior probability that `mu > 0`, the 95% credible interval, the posterior predictive distribution for the next return.

```python
# Probability the strategy has positive expected return
post_mu = trace.posterior["mu"].values.flatten()
print(f"P(mu > 0) = {(post_mu > 0).mean():.2%}")
```

That's the kind of answer Bayesian inference is uniquely good at. No null hypothesis, no test statistic — a direct probability.

## Hierarchical models for cross-sectional strategies

If you have $K$ similar strategies (e.g. variants on a momentum signal across $K$ asset classes), don't fit each independently. Use a hierarchical model that learns the *typical* strategy mean and lets each individual one borrow strength:

```python
with pm.Model() as m:
    mu_population = pm.Normal("mu_pop", 0.0, 0.001)
    sigma_population = pm.HalfNormal("sigma_pop", 0.002)

    mu_k = pm.Normal("mu_k", mu_population, sigma_population, shape=K)
    sigma_k = pm.HalfNormal("sigma_k", 0.02, shape=K)
    nu = pm.Exponential("nu", 1/10) + 2

    pm.StudentT("obs", mu=mu_k[strategy_idx], sigma=sigma_k[strategy_idx],
                nu=nu, observed=stacked_returns)

    trace = pm.sample(2000, tune=1000, chains=4)
```

The hierarchical structure says: "these strategies are different, but related." Each `mu_k` is pulled toward the population mean. A strategy with a few months of great data has its mean shrunk substantially; a strategy with five years has barely any shrinkage. The math handles the balance automatically.

This is the right model for **regime overlays across asset classes** or for **factor variants** that share a parent.

## Bayesian regime detection — Hidden Markov Models

A discrete latent state $z_t \in \{1, ..., K\}$ that evolves with a transition matrix, emitting observations under state-conditional distributions. The classic application: "is the market in a calm or volatile regime?"

```python
from hmmlearn.hmm import GaussianHMM

X = realised_vol.values.reshape(-1, 1)
model = GaussianHMM(n_components=2, covariance_type="full", n_iter=200)
model.fit(X)
regime = model.predict(X)
```

This is empirical Bayes (the MLE-based variant). For full Bayes with PyMC, see the PyMC docs on `pm.HiddenMarkovModel`. Module 16 (Production Strategies) wires this into a real regime-overlay strategy.

## Sequential updating

The killer Bayesian feature for live trading: **the posterior of yesterday is the prior of today**. Update incrementally as data arrives:

```python
# Toy example: estimating probability of a hit-rate parameter
a, b = 1, 1           # Beta(1, 1) = uniform prior

for trade_outcome in stream:        # each is 1 (win) or 0 (loss)
    if trade_outcome == 1:
        a += 1
    else:
        b += 1
    p_mean = a / (a + b)
    ci = beta.interval(0.95, a, b)
    log(f"win_rate posterior: mean={p_mean:.2%}, CI={ci}")
```

For Kalman-filter-style continuous updating, see Module 7 (Time Series) — that's the linear-Gaussian special case where everything is closed-form.

## The cost of being Bayesian

You will be **slow** (MCMC is iterative) and **opinionated** (the prior matters). For inference that's run once during research, both are fine. For an inner loop of an HFT system, no.

The Bayesian / frequentist debate is mostly settled in practice: use whichever framework gives the cleanest answer to the question you have. For "what should I believe about this parameter?" Bayes is direct. For "how would this estimator behave under repeated sampling?" frequentist tools are the natural language.

## Bottom line

The patterns:

- **Shrinkage of strategy means** — never trust a noisy mean; combine with a prior.
- **Hierarchical pooling across variants** — borrow strength across related models.
- **Sequential updates during live trading** — posterior becomes the next day's prior.
- **PyMC for any model where the math isn't closed-form.**

For the rest of the course, we'll use Bayesian thinking *informally* (shrinkage, posterior predictives) and reach for PyMC explicitly when the cost of MCMC is justified — typically in Modules 11 (Modern ML) and 16 (Production Strategies).

Continue to **[Copulas and tail dependence](04-copulas.md)**.
