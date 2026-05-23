# Diffusion-augmented scenario backtesting

A backtest gives you P&L on the historical path. The historical path is **one sample** from the distribution of "possible histories." For tail risk assessment, you want to know how the strategy behaves across *many* plausible historical alternatives — including ones that never happened but plausibly could have.

Diffusion models (Module 12 chapter 6) generate synthetic financial paths with realistic distributional properties. Combined with backtesting, they let you stress-test strategies against many synthetic crises, regimes, and bull markets.

## The pattern

```
historical bars ──► diffusion model training
                            │
                            ▼
              generate 1000 synthetic paths
                            │
                            ▼
         backtest strategy on each synthetic path
                            │
                            ▼
          distribution of Sharpes / max drawdowns
                            │
                            ▼
              tail risk + confidence interval
```

The win: you go from "the strategy had Sharpe 0.8 on history" to "across 1000 synthetic paths, the strategy's Sharpe has a 95% CI of [0.4, 1.2] and the 5th-percentile max drawdown is -25%". The latter is what you actually want for risk management.

## Training a path diffusion model

We covered the mechanics in Module 12. The recipe for trading paths:

1. Collect historical 60-day return windows.
2. Standardise (zero mean, unit variance per window).
3. Train a diffusion model whose output is a 60-bar return path.
4. At inference, sample new 60-bar paths from the model.

The trained model produces paths with:

- Realistic autocorrelation in absolute returns (vol clustering).
- Approximately correct unconditional distribution.
- Plausible "shapes" (trends, ranges, breakouts).

## A worked example

```python
# Pretend we have a trained diffusion model from Module 12
import numpy as np

class TrainedDiffusion:
    """Mock: in reality, you'd load a trained PyTorch diffusion model."""
    def sample(self, n_paths: int, dim: int) -> np.ndarray:
        # Generate IID Gaussian for demo; real diffusion gives realistic paths
        rng = np.random.default_rng(0)
        # Simulate volatility clustering with AR(1) on absolute returns
        out = []
        for _ in range(n_paths):
            r = np.zeros(dim)
            sigma = 0.01
            for t in range(dim):
                sigma = 0.95 * sigma + 0.05 * abs(rng.normal(0, 0.01))
                r[t] = rng.normal(0, sigma)
            out.append(r)
        return np.stack(out)


def scenario_backtest_distribution(strategy_fn, diffusion_model, n_paths: int = 1000,
                                     path_len: int = 252):
    """Backtest `strategy_fn` on n_paths synthetic paths.
    `strategy_fn(returns: np.ndarray) -> Sharpe: float`."""
    paths = diffusion_model.sample(n_paths=n_paths, dim=path_len)
    sharpes = np.array([strategy_fn(r) for r in paths])
    drawdowns = np.array([_max_drawdown(r) for r in paths])
    return {
        "sharpe_mean": float(sharpes.mean()),
        "sharpe_std": float(sharpes.std()),
        "sharpe_ci_5_95": tuple(np.percentile(sharpes, [5, 95]).tolist()),
        "drawdown_mean": float(drawdowns.mean()),
        "drawdown_ci_5_95": tuple(np.percentile(drawdowns, [5, 95]).tolist()),
    }


def _max_drawdown(returns: np.ndarray) -> float:
    eq = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1).min())


def simple_strategy_sharpe(returns: np.ndarray) -> float:
    # A trivial "always long" strategy
    pnl = returns
    return float(pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 else 0.0
```

Running this gives you a robust distribution of likely Sharpes — not just the one number from history.

## Conditional sampling for stress tests

A more advanced use: condition the diffusion on a known feature (e.g., "VIX = 40" for crisis stress). The diffusion samples paths that look like the historical paths *given* VIX = 40.

For trading, this lets you ask: "How does my strategy perform in a 2008-like regime, even if the strategy didn't exist in 2008?"

```python
# Conditional diffusion is implemented in some libraries (e.g., HuggingFace diffusers).
# The pseudocode:
def conditional_paths(diffusion_model, condition: dict, n_paths: int):
    return diffusion_model.sample_conditional(condition, n_paths=n_paths)


crisis_paths = conditional_paths(model, condition={"vix": 40}, n_paths=500)
crisis_dist = scenario_backtest_distribution(my_strategy, lambda n, d: crisis_paths, n_paths=500)
```

The CI on the strategy under the conditional distribution is your crisis stress test.

## Counterfactual scenarios

What if QE never happened? What if the 2020 crash had been deeper? Conditional diffusion lets you simulate alternate histories — useful for portfolio construction that needs to be robust to regimes that haven't yet occurred.

The cost is honesty about the assumptions: the diffusion model was trained on actual history. Its conditional distribution at "VIX = 100" is extrapolation, not interpolation.

## Risk metrics from the distribution

```python
def comprehensive_stress_report(strategy_fn, diffusion_model, n_paths: int = 1000):
    paths = diffusion_model.sample(n_paths=n_paths, dim=252)
    metrics = {
        "sharpe": [],
        "max_dd": [],
        "var_99": [],
        "cvar_99": [],
        "skew": [],
        "kurt": [],
    }
    for r in paths:
        pnl = strategy_fn_returns(r)        # strategy returns the per-bar pnl, not Sharpe
        if pnl.std() == 0:
            continue
        metrics["sharpe"].append(pnl.mean() / pnl.std() * np.sqrt(252))
        metrics["max_dd"].append(_max_drawdown(pnl))
        metrics["var_99"].append(np.percentile(pnl, 1))
        sorted_pnl = np.sort(pnl)
        metrics["cvar_99"].append(sorted_pnl[:max(1, len(sorted_pnl) // 100)].mean())
    return {k: {"mean": np.mean(v), "5%": np.percentile(v, 5), "95%": np.percentile(v, 95)}
            for k, v in metrics.items()}
```

This gives you a complete distribution of every risk metric you care about — useful for setting position limits, capital allocations, drawdown stops.

## Validating the diffusion model

Before trusting the stress test, verify the diffusion model produces plausible paths:

- **Marginal distribution** — histogram of single-day returns should look like the historical one.
- **Autocorrelation of |returns|** — should show clustering decay similar to historical.
- **Cross-correlation across symbols** (if multivariate) — should match historical structure.
- **Tail behaviour** — 99% and 1% tails should be similar (or honestly wider than) historical.

If any of these fail, the stress test is unreliable.

## Pitfalls

!!! warning "Diffusion can't generate genuinely new regimes"
    The model was trained on history. Its samples are distributions over "things like history". A black swan that has no analogue in training data won't appear.

!!! warning "Mode collapse"
    A poorly-trained diffusion can collapse to a few "modes" — all samples look similar. Always check sample diversity (pairwise distances, density estimates).

!!! warning "Conditional sampling is more fragile"
    The model's conditional distribution at out-of-distribution conditions is unreliable. Don't extrapolate to conditions far from training.

!!! warning "Simulator-trained strategy then live-deployed"
    A strategy that does great on synthetic data but poorly on real history is overfit to the simulator. The stress test is a complement to historical backtest, not a replacement.

## Bottom line

Diffusion-augmented scenario backtesting:

- Turns a single historical Sharpe into a **distribution** over plausible alternative histories.
- Enables **conditional stress testing** at specified regimes.
- Requires a **trained diffusion model** that has been validated against historical statistics.
- Is **complementary to traditional backtesting**, not a substitute.

For institutional risk management, this is the right way to size position limits — based on the distribution of bad-but-plausible outcomes, not the single observed history.

Continue to **[Meta-learning (MAML) for fast adaptation across symbols](07-meta-learning.md)**.
