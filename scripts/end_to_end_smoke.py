"""End-to-end integration smoke test.

Exercises the full pipeline taught in Modules 0, 5, 9, 10:

    universal Feed adapter  →  features  →  labels
                                  ↓
                          purged k-fold CV
                                  ↓
                     GradientBoostingClassifier
                                  ↓
                      vectorised backtest
                                  ↓
                         stats + assertions

Pass ``--no-network`` to run against a synthetic random walk instead of
``yfinance``. CI uses ``--no-network`` so the build never depends on
Yahoo's mood; locally, run without the flag for a real-data check.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from engine.backtest import PurgedKFold, vectorized_backtest
from engine.features import frac_diff


def synthetic_close(n: int = 2_500, seed: int = 0) -> pd.Series:
    """A zero-drift log-normal random walk; produces a roughly balanced
    binary label distribution."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2014-01-02", periods=n, freq="B", tz="UTC")
    return pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.012, n))),
        index=idx,
        name="close",
    )


def yahoo_close(
    symbol: str = "SPY", start: str = "2014-01-01", end: str = "2024-12-31"
) -> pd.Series:
    from engine.data import YFinanceFeed

    return YFinanceFeed().bars(symbol, start, end)["close"]


def build_features(close: pd.Series) -> pd.DataFrame:
    ret = close.pct_change()
    return pd.DataFrame(
        {
            "ret_1": ret,
            "ret_5": close.pct_change(5),
            "ret_21": close.pct_change(21),
            "vol_20": ret.rolling(20).std(),
            "dist_ma200": close / close.rolling(200).mean() - 1,
            "fracdiff_04": frac_diff(close, d=0.4, thresh=1e-2),
        }
    )


def run(close: pd.Series, horizon: int = 5) -> dict:
    returns = close.pct_change().fillna(0.0)
    X = build_features(close)
    fwd = close.pct_change(horizon).shift(-horizon)
    y = (fwd > 0).astype(int).rename("y")
    df = X.join(y).dropna()
    if df.empty:
        raise RuntimeError("empty feature/label dataframe; check inputs")
    X_use, y_use = df.drop(columns="y"), df["y"]

    cv = PurgedKFold(n_splits=5, purge=horizon, embargo=horizon)
    preds = pd.Series(np.nan, index=y_use.index)
    for tr, te in cv.split(len(X_use)):
        if len(np.unique(y_use.iloc[tr])) < 2:
            continue  # skip degenerate fold
        model = GradientBoostingClassifier(
            n_estimators=50, max_depth=3, subsample=0.8, random_state=0
        )
        model.fit(X_use.iloc[tr], y_use.iloc[tr])
        preds.iloc[te] = model.predict_proba(X_use.iloc[te])[:, 1]

    signal_raw = 2 * preds - 1
    signal = signal_raw.where(signal_raw.abs() > 0.10, 0.0)
    positions = signal.reindex(returns.index).fillna(0.0)
    res = vectorized_backtest(returns, positions, cost_per_unit_turnover=3e-4)
    return res.stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-network", action="store_true", help="use synthetic data only")
    parser.add_argument("--symbol", default="SPY", help="Yahoo symbol when --no-network is omitted")
    args = parser.parse_args()

    if args.no_network:
        print("[smoke] mode: synthetic")
        close = synthetic_close()
    else:
        print(f"[smoke] mode: yfinance / {args.symbol}")
        try:
            close = yahoo_close(args.symbol)
        except Exception as exc:  # pragma: no cover
            print(f"[smoke] yfinance failed ({exc!r}); falling back to synthetic")
            close = synthetic_close()

    stats = run(close)
    print("[smoke] backtest stats:")
    for key, value in stats.items():
        print(f"  {key}: {value!r}")

    # Sanity assertions
    assert stats["n_days"] > 100, "too few days"
    assert np.isfinite(stats["sharpe"]) or stats["sharpe"] == 0, "non-finite Sharpe"
    assert stats["max_drawdown"] <= 0, "max drawdown should be non-positive"
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
