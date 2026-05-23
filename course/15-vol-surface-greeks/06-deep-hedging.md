# Deep hedging

Classical option hedging (Black-Scholes delta hedge) is optimal under three assumptions: continuous trading, no transaction costs, and known volatility dynamics. Real markets satisfy *none* of these. **Deep hedging** (Bühler, Gonon, Teichmann, Wood, 2019) is a framework that trains a neural-network hedging policy directly against any specified market model and cost structure, with no requirement to be analytically optimal.

For an options market-maker with realistic costs, deep hedging often beats delta-hedging by 20-40% on hedging-cost variance. This chapter explains why and shows a working sketch.

## The classical baseline

You sold a European call. To hedge:

1. Compute Black-Scholes delta.
2. Hold $\Delta$ shares of underlying.
3. Re-hedge daily (or more often) as $\Delta$ changes.
4. At expiry: sell underlying at $S_T$, settle the call at $\max(S_T - K, 0)$.

Variance of P&L over the hedging period:

- Goes to zero as re-hedging frequency → infinity (perfect hedging, no costs).
- In reality: increases with transaction costs (more re-hedges = more spread paid).

The trade-off: hedge less often → larger gamma P&L variance; hedge more often → larger cost. The optimal frequency depends on the cost structure.

## The deep hedging idea

Train a neural network $\pi_\theta(s, t, \text{position})$ that outputs **the hedge action at each time step**. Minimise the variance (or any risk measure — CVaR, expected shortfall) of the terminal P&L under transaction costs.

```python
import torch
import torch.nn as nn
import numpy as np

class HedgePolicy(nn.Module):
    """Maps (S, t_remaining, current_position) → new position."""
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, s, t_rem, position):
        return self.net(torch.stack([s, t_rem, position], dim=-1)).squeeze(-1)


def simulate_path(S0, T, mu, sigma, n_steps, rng):
    dt = T / n_steps
    z = rng.standard_normal(n_steps)
    log_S = np.log(S0) + np.cumsum((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    return np.concatenate([[S0], np.exp(log_S)])


def deep_hedge_loss(policy, n_paths, K, S0, T, mu, sigma, r,
                    n_steps=50, cost_bps=5.0, seed=0):
    rng = np.random.default_rng(seed)
    paths = np.stack([simulate_path(S0, T, mu, sigma, n_steps, rng) for _ in range(n_paths)])
    paths_t = torch.from_numpy(paths.astype("float32"))
    cash = torch.zeros(n_paths)
    position = torch.zeros(n_paths)
    dt = T / n_steps
    for step in range(n_steps):
        s = paths_t[:, step]
        t_rem = torch.full((n_paths,), (T - step * dt))
        target_position = policy(s, t_rem, position)
        # Transaction cost on position change
        trade = target_position - position
        cost = (cost_bps * 1e-4) * torch.abs(trade) * s
        cash = cash * np.exp(r * dt) - trade * s - cost
        position = target_position
    # Settle the position and the call payoff at expiry
    S_T = paths_t[:, -1]
    cash = cash + position * S_T - torch.clamp(S_T - K, min=0.0)
    pnl = cash
    return pnl.var()                    # variance of P&L → minimise


def train_deep_hedger(K=100, S0=100, T=0.25, sigma=0.20, r=0.04,
                      epochs=200, batch=512):
    policy = HedgePolicy()
    optim = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    for ep in range(epochs):
        loss = deep_hedge_loss(policy, batch, K, S0, T, mu=r, sigma=sigma, r=r, seed=ep)
        optim.zero_grad(); loss.backward(); optim.step()
        if ep % 20 == 0:
            print(f"epoch {ep}: loss={loss.item():.4f}")
    return policy
```

The network learns to hedge each path's position adaptively. Under transaction costs, the optimal policy is NOT the Black-Scholes delta — it hedges less aggressively when spot is far from the strike (low gamma, low expected P&L change) and more aggressively when near.

## What the policy learns

Compare the learned policy to Black-Scholes delta:

- In **calm regions** (spot far from strike), the policy holds positions for longer to avoid costs.
- In **gamma-heavy regions** (spot near strike, late in life), the policy re-hedges aggressively despite costs because the variance reduction pays for them.
- The policy is **path-independent** by construction (it only sees current state), so it generalises across paths.

## Beyond plain delta — risk-aware objectives

The natural loss isn't just variance. CVaR (Module 6, chapter 5) or "expected shortfall on the worst 5% of paths" gives a hedger that's robust to tail losses. Plug in any differentiable risk functional.

```python
def cvar_loss(pnl, alpha=0.95):
    """Mean of the worst (1 - alpha) fraction of pnl."""
    threshold = torch.quantile(pnl, 1 - alpha)
    tail = pnl[pnl <= threshold]
    return -tail.mean()    # negative because pnl is positive
```

Replace `pnl.var()` with `cvar_loss(pnl)` in the training loop and the hedger learns to avoid tail losses, not just minimise variance.

## Why this beats classical delta-hedging under costs

Black-Scholes delta is **optimal under continuous, frictionless hedging**. Under discrete trading with costs, it **over-hedges** — re-hedging too often, paying too much in spreads. The deep hedger learns the right trade-off automatically.

Empirical results from the Bühler-Teichmann-Wood paper (and subsequent literature): 20-40% improvement in hedging variance for vanilla options under realistic cost structures. For exotics (cliquets, autocallables), the gains can be larger.

## What deep hedging is NOT good at

- **It needs a forward-simulating market model.** The training environment must simulate the underlying. If the model is wrong (e.g., real markets jump and the simulator doesn't), the hedger learns the wrong policy.
- **It's sample-hungry.** Tens of thousands of paths per epoch; many epochs.
- **It's opaque.** The learned policy is a neural network; you can't easily prove its risk properties.

For production use, the typical recipe: train deep hedging on a realistic simulator (Heston + jumps + costs), backtest the policy against historical replay, paper-trade for months, then deploy in shadow mode alongside delta hedging before promoting.

## Pitfalls

!!! warning "Simulator bias"
    A deep hedger trained on a too-simple model (pure GBM with no jumps) will under-hedge real jumps. Use a Bates model or empirical bootstrap of historical returns as the training distribution.

!!! warning "Stationary distribution assumption"
    The policy is trained for one underlying distribution. Regime shifts (vol cycles) require retraining or domain randomisation across regimes.

!!! warning "Path independence in state"
    The simplest deep hedger only sees current spot, time, position. For path-dependent options or for capturing momentum effects, include lagged spot in the state.

!!! warning "Stability"
    Deep hedging training is sometimes unstable. Use gradient clipping, smaller learning rates for the variance loss (which has bad scale), and multiple training seeds.

## Bottom line

Deep hedging is the right framework when:

- You're hedging options under **non-trivial transaction costs**.
- You're hedging **exotic options** with non-trivial path dynamics.
- You're willing to **trust a simulator** to train against.

For vanilla options under tight markets and low costs, classical delta-hedging is good enough. For everything else, deep hedging is the modern answer.

## End of Module 15

You now have the complete options machinery: pricing (Module 14 + Heston/local vol here), Greeks, IV solvers, surface parameterisations (SVI, SABR, Heston), dealer-flow analysis, and the deep-hedging frontier. The next module — **Production Strategies** — finally puts everything together: HMM regime overlays, Kalman pairs, gradient-boosted classifiers with conformal sizing, vol-targeted carry, earnings-drift strategies, calendar/diagonal spread strategies driven by IV term-structure.

Continue to **[Module 16 — Production Strategies](../16-production-strategies/index.md)**.
