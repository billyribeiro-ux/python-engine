"""Vectorised single-asset backtester.

Designed for daily-or-coarser strategies on a single asset. Inputs are a
returns Series and a positions Series (in units of "fraction of capital").
The engine applies a one-bar shift to the positions (no look-ahead), charges
configurable per-unit-turnover slippage, and returns a tidy P&L series and a
statistics dict.

Anything more sophisticated (event-driven, multi-asset with explicit fills,
broker constraints) lives in ``engine.backtest.event_driven``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class BacktestResult:
    pnl: pd.Series  # net return per bar
    equity: pd.Series  # cumulative net wealth, starting from 1.0
    turnover: pd.Series  # |Δposition| per bar
    positions: pd.Series  # the actually-applied (shifted) positions

    @property
    def stats(self) -> dict[str, float]:
        r = self.pnl.dropna()
        if r.empty:
            return {
                "sharpe": float("nan"),
                "ann_return": 0.0,
                "ann_vol": 0.0,
                "max_drawdown": 0.0,
                "calmar": float("nan"),
                "n_days": 0,
                "turnover_ann": 0.0,
            }
        ann_factor = 252  # daily-bar default; override for other intervals
        ann_ret = r.mean() * ann_factor
        ann_vol = r.std(ddof=1) * np.sqrt(ann_factor)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
        equity = self.equity
        peak = equity.cummax()
        dd = (equity / peak - 1.0).min()
        calmar = ann_ret / abs(dd) if dd < 0 else float("nan")
        return {
            "sharpe": float(sharpe),
            "ann_return": float(ann_ret),
            "ann_vol": float(ann_vol),
            "max_drawdown": float(dd),
            "calmar": float(calmar),
            "n_days": len(r),
            "turnover_ann": float(self.turnover.sum() * ann_factor / len(r)),
        }


def vectorized_backtest(
    returns: pd.Series,
    positions: pd.Series,
    cost_per_unit_turnover: float = 0.0,
    shift_signal: bool = True,
) -> BacktestResult:
    """Run a vectorised backtest.

    Parameters
    ----------
    returns : pd.Series
        Per-bar simple returns of the underlying asset.
    positions : pd.Series
        Target position per bar, in units of "fraction of capital". Aligned
        to ``returns`` by index; missing values become 0.
    cost_per_unit_turnover : float
        Linear transaction cost charged per absolute change in position.
        For 2bps per turn (e.g. 1bp half-spread + 1bp commission) pass
        ``2e-4``.
    shift_signal : bool
        If True (default), positions are shifted one bar forward so today's
        position acts on tomorrow's return. **Always True for honest
        backtests** — turning it off is for diagnostic comparisons only.

    Returns
    -------
    BacktestResult
    """
    pos_aligned = positions.reindex(returns.index).fillna(0.0)
    pos_traded = pos_aligned.shift(1).fillna(0.0) if shift_signal else pos_aligned
    turnover = pos_traded.diff().abs().fillna(pos_traded.abs())
    gross = pos_traded * returns.fillna(0.0)
    cost = cost_per_unit_turnover * turnover
    pnl = gross - cost
    equity = (1.0 + pnl).cumprod()
    return BacktestResult(pnl=pnl, equity=equity, turnover=turnover, positions=pos_traded)
