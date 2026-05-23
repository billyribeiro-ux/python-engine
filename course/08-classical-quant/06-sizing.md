# Sizing — Kelly, fractional Kelly, CVaR

You have a strategy with an expected edge and a known risk distribution. How much capital should you put on each bet? This is the **sizing** problem, and it is roughly as important as having an edge at all — a 0.5 Sharpe strategy sized correctly outperforms a 1.0 Sharpe strategy sized too aggressively over any meaningful horizon.

## The Kelly criterion — the optimal-growth fraction

Kelly (1956) asked: given a sequence of bets with a positive expected log-return, what fraction $f$ of bankroll on each bet maximises the long-run growth rate?

For a continuous-return setting with expected return $\mu$ and variance $\sigma^2$ (per period):

$$
f^* = \frac{\mu}{\sigma^2}
$$

For a strategy with Sharpe $S = \mu / \sigma$:

$$
f^* = \frac{S}{\sigma}
$$

i.e., the Kelly fraction scales as the Sharpe over the volatility.

The growth rate at Kelly is:

$$
g_{\text{Kelly}} = \tfrac{1}{2} S^2
$$

A strategy with Sharpe 1 grows at $\frac{1}{2} \cdot 1^2 = 0.5$ per period at full Kelly — that's a tremendous return.

## Why nobody uses full Kelly

Three reasons full Kelly is too aggressive in practice:

1. **You don't know $\mu$ exactly.** Full Kelly assumes you do. Overestimating $\mu$ by 50% means you size at 1.5× Kelly, which is *past the peak* of the growth curve — you start losing money.
2. **Drawdowns at full Kelly are brutal.** The expected drawdown at full Kelly is **the entire bankroll**, with non-trivial probability. Most institutional investors will fire you at 30%.
3. **Estimation error compounds.** Even with a true Sharpe of 1, sample Sharpes for short backtests range widely. Sizing on the sample → over-sizing the strategies that got lucky.

The institutional consensus: **half-Kelly or less**. A common rule is to size at 25-50% of Kelly — gives up about 1/4 of the growth rate, halves the volatility, reduces expected max drawdown enormously.

```python
import numpy as np

def kelly_fraction(mu: float, sigma: float, k: float = 0.5) -> float:
    """Kelly fraction with a `k` multiplier (k=0.5 is half-Kelly)."""
    return k * mu / (sigma ** 2)
```

## Vol-targeting — Kelly's pragmatic cousin

Most quant shops don't compute Kelly explicitly. They **vol-target**: size positions to a fixed annualised volatility, regardless of strategy.

```python
def vol_target_size(strategy_returns: pd.Series, target_vol: float = 0.10,
                    lookback: int = 60) -> pd.Series:
    realised = strategy_returns.rolling(lookback).std() * np.sqrt(252)
    return (target_vol / realised).clip(upper=3.0).shift(1)
```

Equivalence: vol-targeting at $\sigma^*$ is *exactly* the same as Kelly sizing if the Sharpe is held constant. The two are dual formulations of the same problem.

Vol-targeting is preferred in practice because:
- It's interpretable ("we run at 10% vol").
- It doesn't require an explicit edge estimate.
- It naturally de-risks during high-vol regimes.

## CVaR-targeting — the right thing for fat-tailed strategies

For strategies whose return distribution is fat-tailed (carry, short-vol, illiquid alts), vol underestimates the real downside. Target CVaR instead:

```python
import numpy as np
from scipy.stats import genpareto

def cvar_target_size(returns: np.ndarray, capital: float, cvar_budget: float,
                     alpha: float = 0.99) -> float:
    losses = -returns
    u = np.quantile(losses, 0.95)
    excess = losses[losses > u] - u
    shape, _, scale = genpareto.fit(excess, floc=0)
    p_exceed = (losses > u).mean()
    var = u + scale / shape * (((1 - alpha) / p_exceed) ** (-shape) - 1)
    cvar = (var + scale - shape * u) / (1 - shape)
    return cvar_budget / cvar * capital
```

If your strategy's 99% CVaR is 4% of capital and you've set a budget of 2%, this scales the strategy down to half.

For short-vol strategies in particular, **CVaR targeting catches the 2018-Feb-style blow-up risk that vol-targeting doesn't**, because vol-targeting de-risks *after* vol spikes, not before.

## Sizing a portfolio of strategies

If you have $K$ strategies, each with vol-targeted PnL, the natural combination is **inverse-volatility weighted**:

$$
w_k = \frac{1/\sigma_k}{\sum_j 1/\sigma_j}
$$

For strategies with similar Sharpe, this is approximately Markowitz-optimal *and* much more robust to estimation error.

If Sharpes differ, scale by Sharpe over vol:

$$
w_k \propto \frac{S_k}{\sigma_k}
$$

Cap individual weights (e.g. no strategy > 30%) and rescale.

## A worked example: end-to-end sizing for a TSM portfolio

```python
from engine.data import YFinanceFeed, ParquetCache
import numpy as np
import pandas as pd

feed = ParquetCache(YFinanceFeed(), root="data/bars")
universe = ["SPY", "QQQ", "TLT", "GLD", "EFA"]
prices = pd.concat(
    {s: feed.bars(s, "2014-01-01", "2024-12-31")["close"] for s in universe},
    axis=1,
).dropna()
returns = prices.pct_change().dropna()

# TSM signal: long if above 200d MA, short below
signal = (prices > prices.rolling(200).mean()).astype(int) * 2 - 1
signal = signal.shift(1).reindex(returns.index).fillna(0)

# Per-asset positions: signal scaled by inverse vol, 10% target per asset
per_asset_vol = returns.rolling(60).std() * np.sqrt(252)
size = (0.10 / per_asset_vol).clip(upper=3.0)
positions = signal * size

# Per-asset PnL
asset_pnl = positions.shift(1) * returns
portfolio_pnl = asset_pnl.mean(axis=1)        # equal weight across the 5 strategies

# Vol-target the whole portfolio
port_vol = portfolio_pnl.rolling(60).std() * np.sqrt(252)
scale = (0.12 / port_vol).clip(upper=3.0)
final_pnl = portfolio_pnl * scale.shift(1)

ann_sharpe = final_pnl.mean() / final_pnl.std() * np.sqrt(252)
print(f"TSM portfolio Sharpe (gross): {ann_sharpe:.2f}")
```

Two layers of vol targeting (per-asset and portfolio-level) give a remarkably stable equity curve. Sharpe ~0.7-0.9 gross is typical; net of costs and slippage, ~0.5-0.7.

## Drawdown stops

Beyond sizing, you want an explicit **drawdown stop**:

```python
def drawdown_stop(pnl: pd.Series, max_drawdown: float = 0.20) -> pd.Series:
    """Returns a multiplier in [0, 1] to scale future positions."""
    equity = (1 + pnl).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1
    # As DD approaches max_drawdown, scale down linearly to zero
    factor = (1 - (-dd / max_drawdown).clip(0, 1)) ** 2
    return factor.shift(1).fillna(1.0)
```

When you're at 20% drawdown, your sizing factor is zero — you're flat. Below that, it scales smoothly. This is more graceful than a hard kill switch and lets the strategy recover without re-entering the full position immediately.

## Pitfalls

!!! warning "Sizing on in-sample Sharpe"
    Your in-sample Sharpe is upward-biased by selection. Use a deflated Sharpe (Module 6) or shrunk Sharpe for sizing.

!!! warning "Vol-targeting a strategy with regime breaks"
    Vol-targeting reacts *after* vol changes. For strategies prone to abrupt regime breaks (carry, vol-selling), supplement with hard hard exposure caps and CVaR targeting.

!!! warning "Compounding vs. constant-dollar"
    Be explicit about whether your sizing scales with capital or is constant. The two have very different drawdown characteristics — constant-dollar drawdowns recover linearly, compounding drawdowns recover exponentially (and look worse on a log scale).

!!! warning "Kelly on multiple correlated strategies"
    The naive sum-of-Kelly's over-allocates when strategies are correlated. The right answer is multivariate Kelly, which reduces to a Markowitz-style optimisation.

## Bottom line

For sizing:

- **Half-Kelly or less** — full Kelly is too aggressive for any sample-estimated edge.
- **Vol-target your strategies at 8-12% annual** — interpretable, regime-adaptive, equivalent to half-Kelly for typical Sharpes.
- **CVaR-target for fat-tailed strategies** — vol underestimates the real downside.
- **Drawdown stops as an outer-loop safety** — even with good sizing, things go wrong.
- **Inverse-vol weight across a portfolio of strategies**, cap per-strategy weight.

## End of Module 8

You now have the bedrock strategies and the sizing layer they sit on. Module 9 is next: **backtesting that doesn't lie**. The work in Module 8 will look great in a naive backtest. The work in Module 9 is what makes the backtest match live trading.

Continue to **[Module 9 — Backtesting](../09-backtesting/index.md)**.
