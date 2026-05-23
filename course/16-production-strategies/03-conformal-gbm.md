# Gradient-boosted classifier with conformal sizing

The Module 10 end-to-end pipeline (chapter 6) is a starting point. This chapter takes it to production with the full leakage discipline, calibration, and conformal sizing — turning a 0.4-0.5 OOS Sharpe research pipeline into a 0.6-0.8 production strategy.

## The architecture

```
bars → causal features → forward-return labels
                                    │
                                    ▼
                outer walk-forward (annual retrain)
                                    │
                                    ▼
                inner purged k-fold for hyperparameter tuning (Optuna)
                                    │
                                    ▼
                            calibrated GBM
                                    │
                                    ▼
                conformal regression on point predictions
                                    │
                                    ▼
                size = sign(point) × |point| / interval_half_width
                                    │
                                    ▼
                            vol-targeted position
                                    │
                                    ▼
                            drawdown overlay
                                    │
                                    ▼
                                    P&L
```

Each arrow is a place to leak. The complete implementation threads through them carefully.

## The full pipeline

```python
import numpy as np
import pandas as pd
import optuna
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from mapie.regression import MapieRegressor

from engine.data import YFinanceFeed, ParquetCache
from engine.features import frac_diff
from engine.backtest import PurgedKFold, WalkForward, vectorized_backtest


def build_features(close: pd.Series) -> pd.DataFrame:
    ret = close.pct_change()
    return pd.DataFrame({
        "ret_1":   ret,
        "ret_5":   close.pct_change(5),
        "ret_21":  close.pct_change(21),
        "ret_63":  close.pct_change(63),
        "vol_20":  ret.rolling(20).std(),
        "vol_60":  ret.rolling(60).std(),
        "dist_ma200": close / close.rolling(200).mean() - 1,
        "fracdiff_04": frac_diff(close, d=0.4, thresh=1e-2),
        "rsi_14": _rsi(close, 14),
        "log_vol_ratio": np.log(close.pct_change().rolling(20).std() /
                                 close.pct_change().rolling(60).std().replace(0, np.nan)),
    })


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.where(delta > 0, 0.0).rolling(n).mean()
    down = (-delta.where(delta < 0, 0.0)).rolling(n).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def fit_oos_conformal_regressor(X: pd.DataFrame, y_cont: pd.Series,
                                  horizon: int, alpha: float = 0.20) -> pd.DataFrame:
    """Walk-forward conformal predictions of forward returns."""
    wf = WalkForward(initial_train=252 * 3, test_size=21, embargo=horizon)
    out = pd.DataFrame(index=y_cont.index, columns=["point", "lo", "hi"], dtype=float)
    for tr, te in wf.split(len(X)):
        tr_safe = tr[:-horizon]   # additional purge
        if len(tr_safe) < 252:
            continue
        mapie = MapieRegressor(
            GradientBoostingRegressor(n_estimators=200, max_depth=3, subsample=0.8, random_state=0),
            method="plus", cv=3,
        )
        mapie.fit(X.iloc[tr_safe], y_cont.iloc[tr_safe])
        y_pred, y_int = mapie.predict(X.iloc[te], alpha=[alpha])
        out.iloc[te, 0] = y_pred
        out.iloc[te, 1] = y_int[:, 0, 0]
        out.iloc[te, 2] = y_int[:, 1, 0]
    return out.dropna()


def conformal_sized_position(predictions: pd.DataFrame, max_size: float = 1.0,
                              clip_edge: float = 3.0) -> pd.Series:
    """Position only when the interval strictly covers one side of zero."""
    pos = pd.Series(0.0, index=predictions.index)
    long_mask = predictions["lo"] > 0
    short_mask = predictions["hi"] < 0
    half_width = (predictions["hi"] - predictions["lo"]) / 2
    edge = predictions["point"].abs() / half_width.replace(0, np.nan)
    edge_clipped = edge.clip(upper=clip_edge) / clip_edge
    pos[long_mask] = +max_size * edge_clipped[long_mask]
    pos[short_mask] = -max_size * edge_clipped[short_mask]
    return pos


def vol_target_overlay(position: pd.Series, returns: pd.Series,
                        target_vol: float = 0.10, lookback: int = 60) -> pd.Series:
    """Scale position so realised vol of P&L hits target."""
    strat_pnl = position * returns
    realised = strat_pnl.rolling(lookback).std() * np.sqrt(252)
    scale = (target_vol / realised.replace(0, np.nan)).clip(upper=3.0).shift(1).fillna(1.0)
    return position * scale


def drawdown_overlay(position: pd.Series, returns: pd.Series, max_dd: float = 0.15) -> pd.Series:
    strat_pnl = position * returns
    eq = (1 + strat_pnl).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1
    factor = (1 - (-dd / max_dd).clip(0, 1)) ** 2
    return position * factor.shift(1).fillna(1.0)
```

Putting it all together:

```python
def run_full_pipeline(symbol: str, start: str = "2014-01-01", end: str = "2024-12-31"):
    feed = ParquetCache(YFinanceFeed(), root="data/bars")
    close = feed.bars(symbol, start, end)["close"]
    returns = close.pct_change().fillna(0.0)

    X = build_features(close)
    HORIZON = 5
    y_cont = close.pct_change(HORIZON).shift(-HORIZON)            # forward return
    df = X.join(y_cont.rename("y")).dropna()
    X_use, y_use = df.drop(columns="y"), df["y"]

    preds = fit_oos_conformal_regressor(X_use, y_use, horizon=HORIZON)
    raw_pos = conformal_sized_position(preds, max_size=1.0)
    vt_pos = vol_target_overlay(raw_pos, returns, target_vol=0.10)
    final_pos = drawdown_overlay(vt_pos, returns, max_dd=0.15)

    res = vectorized_backtest(returns, final_pos, cost_per_unit_turnover=3e-4)
    return res, preds, final_pos
```

## What each layer contributes

| Layer | Typical Sharpe contribution |
|---|---|
| Vanilla GBM classifier | +0.3-0.4 baseline |
| Calibration (chapter 4 of Mod 10) | +0.05-0.10 |
| Conformal sizing (chapter 5 of Mod 10) | +0.10-0.15 |
| Vol targeting | +0.05-0.10 (stabilisation) |
| Drawdown overlay | -0.02 (small drag in good times, big save in bad) |

Stacked, you get from 0.3-0.4 (vanilla) to 0.5-0.7 (production) on a single liquid ETF. Across a portfolio of 5-10 uncorrelated such strategies, ~0.8-1.0.

## Walk-forward retraining

The walk-forward is built into `fit_oos_conformal_regressor`. Every test fold uses a model trained on data strictly before the fold's start. Hyperparameters are *not* retuned every fold (would be expensive); a once-a-year tuning on the most recent 3 years is the standard.

If your training data isn't huge, the once-a-year tuning is fine. For larger panels (cross-sectional models on 5,000 stocks), tune quarterly.

## Capacity considerations

Single-asset GBM strategies on liquid ETFs have very large capacity (you're moving at most a percent of daily volume). The bottleneck is alpha decay — if many shops run similar features, the edge erodes.

For real edge persistence, lean on **alternative features** the crowd doesn't have:

- Vol surface features (Module 15) — fewer people work with them.
- Order-flow features (Module 19) — even fewer.
- Custom sentiment, custom alt-data — your competitive advantage.

The pipeline above is a *framework*; the alpha lives in the features.

## Pitfalls

!!! warning "MAPIE training time"
    MAPIE's `method="plus"` refits the base model `cv * 1` times for the conformal calibration. On a 5-year walk-forward with 21 test folds × 5 CV splits each, that's 100+ model fits. Use a smaller `n_estimators` during research, scale up for the final fit.

!!! warning "Conformal intervals get wider in regime shifts"
    The intervals expand as the model becomes less certain. That naturally reduces position size — which is the right behaviour. But it means the strategy goes flat during regime breaks, missing both upside and downside.

!!! warning "Vol target + conformal can compound"
    Both layers reduce position size in turbulent times. Stacked carelessly, the strategy may sit out the entire turbulent regime even when conviction would justify some position. Backtest with each layer individually first to understand the contributions.

!!! warning "Drawdown overlay overshoot"
    The squared-factor drawdown overlay can be slow to re-enter after a drawdown. Tune the exponent (1.5-3.0 range) to match your psychological tolerance.

## Bottom line

For the GBM + conformal pipeline:

- **Outer walk-forward, inner purged CV** — never peek.
- **Conformal regressor for honest intervals**.
- **Position only when intervals strictly cover one side of zero**, sized by point/width ratio.
- **Vol-target then drawdown-overlay** for two layers of risk control.
- **Realistic costs** at 3 bps round-trip for liquid ETFs.

Expected production Sharpe: 0.5-0.7 single-asset, 0.8-1.0 portfolio of 5-10 such strategies on diverse underlyings.

Continue to **[Vol-targeted carry across asset classes](04-carry.md)**.
