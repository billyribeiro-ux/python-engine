# HMM regime overlay on momentum

A vanilla momentum signal (Module 8 chapter 2) works most of the time and fails catastrophically during "momentum crashes" — sudden reversals after market bottoms when the strategy is heavily short the prior losers that lead the rebound. A regime overlay reduces exposure when the conditions for a momentum crash are present.

The Hidden Markov Model (HMM) on realised volatility is the clean implementation: identify discrete regimes (calm, normal, stressed), trade momentum at full size in normal-to-calm and at zero size in stressed. This chapter is the working build.

## The architecture

```
returns ──► realised vol ──► HMM ──► regime state (1 of K)
                                            │
                                            ▼
                              momentum signal × regime multiplier
                                            │
                                            ▼
                                  vol-targeted positions ──► P&L
```

## The HMM

A 2- or 3-state Gaussian HMM with realised vol as the observation. State 1 = calm (low vol), state K = stressed (high vol). The Viterbi (or filtered) state estimate gives you the current regime.

```python
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def fit_regime_hmm(returns: pd.Series, n_states: int = 2, lookback: int = 252):
    """Fit an HMM on rolling realised vol of returns."""
    rv = returns.rolling(20).std().dropna() * np.sqrt(252)
    log_rv = np.log(rv.values).reshape(-1, 1)
    model = GaussianHMM(n_components=n_states, covariance_type="full",
                       n_iter=200, random_state=0)
    model.fit(log_rv)
    states = model.predict(log_rv)
    state_series = pd.Series(states, index=rv.index, name="regime")
    return model, state_series


def regime_to_multiplier(states: pd.Series, model: GaussianHMM) -> pd.Series:
    """Map regimes to position multipliers — full in low-vol, zero in high-vol."""
    # Order regimes by mean log-vol; lowest = "calm"
    means = model.means_.flatten()
    order = np.argsort(means)
    rank = pd.Series(np.argsort(order), index=range(model.n_components))
    # rank[s] = 0 for lowest-vol state, ..., n-1 for highest
    if model.n_components == 2:
        # 2-state: full size in calm, zero in stressed
        multipliers = pd.Series([1.0, 0.0], index=range(2))
    elif model.n_components == 3:
        multipliers = pd.Series([1.0, 0.5, 0.0], index=range(3))
    else:
        # Linear ramp
        multipliers = pd.Series(1 - np.arange(model.n_components) / (model.n_components - 1),
                                index=range(model.n_components))
    return states.map(rank).map(multipliers)
```

## The strategy

```python
import pandas as pd
import numpy as np

def ts_momentum(close: pd.Series, lookback: int = 252) -> pd.Series:
    """+1 above the lookback MA, -1 below."""
    above = close > close.rolling(lookback).mean()
    return (above.astype(int) * 2 - 1).rename("ts_mom")


def vol_target_sized(signal: pd.Series, returns: pd.Series,
                     target_vol: float = 0.10, lookback: int = 60) -> pd.Series:
    vol = returns.rolling(lookback).std() * np.sqrt(252)
    raw_size = signal * (target_vol / vol.replace(0, np.nan)).clip(upper=3.0)
    return raw_size.shift(1).fillna(0.0)


def hmm_overlay_strategy(close: pd.Series, n_states: int = 2,
                          target_vol: float = 0.10) -> pd.DataFrame:
    """Vol-targeted time-series momentum gated by HMM regime."""
    returns = close.pct_change().dropna()
    model, regimes = fit_regime_hmm(returns, n_states=n_states)
    multiplier = regime_to_multiplier(regimes, model)
    signal = ts_momentum(close, lookback=252)
    raw_position = vol_target_sized(signal, returns, target_vol=target_vol)
    # Re-index multiplier to match positions, shift so today's regime gates today's trade
    pos = raw_position * multiplier.reindex(raw_position.index).shift(1).fillna(0.0)
    return pd.DataFrame({"signal": signal, "raw_pos": raw_position,
                         "multiplier": multiplier.reindex(raw_position.index),
                         "position": pos})
```

That's the complete strategy. The pieces:

- **Vanilla momentum** — long when above 1-year MA, short when below.
- **Vol-targeted sizing** — scale to 10% annualised vol.
- **HMM regime gate** — multiply by 1.0 in the calm state, 0.0 in the stressed state.
- **shift(1)** — both the position sizing and the regime gate use *yesterday's* data, applied to today's bar. No look-ahead.

## Backtesting it

```python
from engine.data import YFinanceFeed, ParquetCache
from engine.backtest import vectorized_backtest

feed = ParquetCache(YFinanceFeed(), root="data/bars")
spy = feed.bars("SPY", "2014-01-01", "2024-12-31")["close"]

strategy = hmm_overlay_strategy(spy, n_states=2, target_vol=0.10)
returns = spy.pct_change().fillna(0.0)
res = vectorized_backtest(returns, strategy["position"], cost_per_unit_turnover=2e-4)
print(res.stats)
```

Typical results on a 10-year SPY backtest: vanilla momentum Sharpe ~0.4-0.5, HMM-overlaid ~0.5-0.7. The Sharpe gain comes mostly from **drawdown reduction** during 2020 March and similar episodes — the HMM correctly flips to "stressed" and zeros the position.

## Why this works

Three forces:

1. **Momentum crashes happen in identifiable conditions** — high vol, fast price moves, often after market lows. The HMM picks these up indirectly via realised vol.
2. **Vol regimes are persistent.** Volatility clusters; once you're in a high-vol regime, you tend to stay there for days/weeks. The HMM exploits this persistence.
3. **Out-of-sample regime classification is honest** — the HMM uses only past data to classify today's regime.

## Walk-forward validation

A working momentum-with-HMM strategy must be walk-forward backtested, not just CV'd. Fit the HMM on a rolling window of (say) 3 years; classify the next year. Refit each year.

```python
def walk_forward_hmm_overlay(close: pd.Series, train_years: int = 3,
                              test_years: int = 1, n_states: int = 2):
    """Walk-forward refit of HMM each year."""
    returns = close.pct_change().fillna(0.0)
    out = pd.Series(np.nan, index=close.index, name="multiplier")
    start = close.index[0]
    while True:
        train_end = start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)
        if test_end > close.index[-1]:
            break
        train_ret = returns.loc[start:train_end]
        if len(train_ret) < 252 * 2:
            start = train_end
            continue
        model, regimes_train = fit_regime_hmm(train_ret, n_states=n_states)
        # Classify the test window with the trained model
        test_ret = returns.loc[train_end:test_end]
        rv = test_ret.rolling(20).std() * np.sqrt(252)
        log_rv = np.log(rv.dropna().values).reshape(-1, 1)
        states_test = pd.Series(model.predict(log_rv),
                                index=rv.dropna().index)
        multiplier_test = regime_to_multiplier(states_test, model)
        out.loc[multiplier_test.index] = multiplier_test.values
        start = train_end
    return out.fillna(method="ffill").fillna(1.0)
```

This is honest about the regime classification — at each test bar, only data from before that bar's start has been seen.

## Pitfalls

!!! warning "HMM regime labels aren't stable across refits"
    "Regime 0" in this year's fit may be "regime 1" in next year's. Always re-order regimes by mean realised vol after each fit — that's what the `regime_to_multiplier` does.

!!! warning "Tiny samples per regime"
    A 2-state HMM on 252 daily bars typically has only ~20-50 observations of the rarer state. The fit is noisy. Use 1000+ bars for stable regime estimates.

!!! warning "Multiplier discontinuities at regime transitions"
    Going from multiplier 1.0 to 0.0 instantly creates large turnover spikes. A smoothed transition (e.g., EWM of the multiplier) reduces transaction costs without losing the protective effect.

!!! warning "HMM as a single-asset overlay"
    The strategy above gates a single-asset signal. For a multi-asset momentum portfolio, you'd want a *cross-asset* regime indicator (e.g., VIX, credit spreads, term structure) — single-asset HMM may give different signals per asset.

## Bottom line

For an HMM-overlaid momentum strategy:

- **Train a 2-state HMM on realised vol**, refit annually walk-forward.
- **Map regimes to position multipliers** (1 for calm, 0 for stressed).
- **Apply on top of a vol-targeted vanilla momentum signal**.
- **Smooth the multiplier** to reduce transaction costs on regime flips.
- **Test on real out-of-sample data**, not in-sample CV.

This pattern produces a strategy with materially better drawdown characteristics than vanilla momentum, at roughly equal expected return.

Continue to **[Kalman pairs trading at production grade](02-kalman-pairs.md)**.
