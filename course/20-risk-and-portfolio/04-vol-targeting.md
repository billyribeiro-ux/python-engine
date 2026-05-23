# Vol targeting at portfolio level

Vol targeting at the **strategy level** (we covered it in Modules 4 and 8) keeps each individual strategy operating near its design vol. **Portfolio-level** vol targeting adds an outer loop that keeps the aggregated portfolio's realised vol within the target. The two layers combined produce remarkably stable equity curves.

## The two layers

```
strategy 1 ────► vol-targeted ──┐
strategy 2 ────► vol-targeted ──┼──► aggregated PnL ──► portfolio vol-target ──► final
strategy 3 ────► vol-targeted ──┘
```

Each strategy's internal vol-target keeps it at (e.g.) 10% vol. The portfolio target keeps the *aggregate* at (e.g.) 12% vol.

Why both? Strategies are imperfectly diversified, so the aggregate's vol drifts with the correlation regime. The outer target stabilises the aggregate.

## The implementation

```python
import numpy as np
import pandas as pd


def vol_target_per_strategy(returns: pd.Series, target_vol: float = 0.10,
                              lookback: int = 60, cap: float = 3.0) -> pd.Series:
    """Scale a strategy's returns so realised vol matches target_vol."""
    rolling_vol = returns.rolling(lookback).std() * np.sqrt(252)
    scale = (target_vol / rolling_vol.replace(0, np.nan)).clip(upper=cap).shift(1).fillna(1.0)
    return returns * scale


def vol_target_portfolio(per_strategy_returns: pd.DataFrame,
                           strategy_target: float = 0.10,
                           portfolio_target: float = 0.12,
                           lookback: int = 60, cap: float = 3.0) -> pd.Series:
    """Two-layer vol targeting: per-strategy then portfolio."""
    # Layer 1
    sized = per_strategy_returns.apply(
        vol_target_per_strategy, target_vol=strategy_target, lookback=lookback, cap=cap
    )
    aggregate = sized.mean(axis=1)
    # Layer 2
    rolling_vol = aggregate.rolling(lookback).std() * np.sqrt(252)
    scale = (portfolio_target / rolling_vol.replace(0, np.nan)).clip(upper=cap).shift(1).fillna(1.0)
    return aggregate * scale
```

For a portfolio of 5-10 reasonably uncorrelated strategies, this delivers a near-constant 12% realised vol — essential for institutional investors who size by vol.

## Choosing the targets

- **Per-strategy target**: matches the design vol of the strategy. For most production strategies, 8-15% annualised.
- **Portfolio target**: depends on investor mandate. Institutional funds-of-funds typically target 8-15%; aggressive long-only equity funds 18-25%; market-neutral pure-alpha shops 6-10%.

The portfolio target should be **less than the sum of strategy targets** — diversification reduces vol below the average.

## When to scale up vs down

When realised vol drops below target:

- Either the strategies are quiet (genuine low-vol regime), or the model thinks they are (high vol elsewhere may be in the future).
- Vol-targeting scales positions UP to maintain target.

When realised vol rises above target:

- A vol spike has happened.
- Vol-targeting scales positions DOWN.

The mechanical scaling reduces drawdowns during vol spikes and amplifies returns during calm regimes. **The net effect on Sharpe is positive** in most regimes, because vol clusters.

## The vol-targeting lag

Vol-targeting is **reactive**: it scales based on **realised** vol (already in the past). For sharp regime changes, the scaling lags by ~$\text{lookback}$ days.

Solutions:

1. **Use a forward-looking vol estimate** — VIX, GARCH forecast (Module 7 chapter 2).
2. **Shorter lookback** — faster but noisier.
3. **EWMA vol** — exponentially-weighted; recent observations matter more.

```python
def ewm_vol_target(returns: pd.Series, target: float = 0.10,
                     halflife: int = 30, cap: float = 3.0) -> pd.Series:
    """EWM-vol-targeted version. Faster response than rolling window."""
    ewm_vol = returns.ewm(halflife=halflife).std() * np.sqrt(252)
    scale = (target / ewm_vol.replace(0, np.nan)).clip(upper=cap).shift(1).fillna(1.0)
    return returns * scale
```

EWM with halflife=30 days is roughly equivalent in responsiveness to a 60-day rolling window but with a smoother profile.

## Vol targeting under regime breaks

In a 2008-March-2020-style regime break, vol-targeting reduces exposure dramatically — sometimes by 80-90%. After the regime stabilises, the portfolio re-grows back to target.

The cost: during the regime break itself, you may miss the rebound. The benefit: you don't go to zero.

Most investors accept this trade-off readily.

## A worked example

```python
# Pretend we have a multi-strategy returns DataFrame
import pandas as pd
import numpy as np

rng = np.random.default_rng(0)
n_days = 1000
strategies = pd.DataFrame({
    "momentum":   rng.normal(0.0003, 0.012, n_days),
    "carry":      rng.normal(0.0002, 0.010, n_days),
    "pairs":      rng.normal(0.0001, 0.005, n_days),
    "calendar":   rng.normal(0.0001, 0.006, n_days),
}, index=pd.date_range("2020-01-01", periods=n_days, freq="B"))

# Inject a vol shock
strategies.iloc[400:430] *= 5

final = vol_target_portfolio(strategies, strategy_target=0.10, portfolio_target=0.12,
                              lookback=60)

print(f"Pre-shock realised vol (avg of strategies): "
      f"{strategies.iloc[:400].mean(axis=1).std() * np.sqrt(252):.2%}")
print(f"Post-shock realised vol (final): "
      f"{final.iloc[400:].std() * np.sqrt(252):.2%}")
```

Pre-shock, the strategies' average vol is around 5% (well diversified). After applying vol targeting, the realised vol comes close to the 12% portfolio target. During the injected shock, the vol-targeting reduces positions significantly.

## Pitfalls

!!! warning "Costs from frequent rebalancing"
    Vol-targeting at daily frequency causes daily turnover even when strategies aren't trading. Limit the rebalance to a threshold (e.g., only rebalance if scale changes by >5%) to save costs.

!!! warning "Vol target on strategies with positive autocorrelation in returns"
    Some strategies have momentum in their PnL (a winning streak begets another). Aggressive vol-targeting "sells the winners" — which is wrong if PnL momentum is real.

!!! warning "Vol target on tail-risky strategies"
    For carry, short-vol, etc., realised vol is a *poor* proxy for tail risk. CVaR-targeting (Module 8 chapter 6) is better.

!!! warning "Scale cap too low"
    A scale cap of 1.0 means you never lever — you can only de-risk. For risk parity to work properly, you need to allow scale > 1.

## Bottom line

Portfolio-level vol-targeting:

- **Two layers**: per-strategy + portfolio.
- **EWM vol** for responsiveness, rolling vol for stability.
- **Scale cap** at 3× to limit leverage risk.
- **Rebalance threshold** to control turnover costs.

The mechanism is mostly mechanical; the cost is small; the drawdown reduction is real.

Continue to **[HRP and NCO — modern hierarchical methods](05-hrp-nco.md)**.
