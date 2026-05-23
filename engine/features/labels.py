"""Trading-specific label constructors.

Implements the López de Prado "triple-barrier" labels and the meta-labeling
helper used by Module 10 and Module 11 of the course.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(
    close: pd.Series,
    events: pd.Index,
    pt_sl: tuple[float, float],
    horizon_bars: int,
    side: pd.Series | None = None,
) -> pd.DataFrame:
    """Triple-barrier labels: profit-take, stop-loss, vertical (time).

    For each event timestamp, walk forward up to ``horizon_bars`` bars and
    record which barrier is hit first.

    Parameters
    ----------
    close : pd.Series
        Close price series, sorted by time.
    events : pd.Index
        Timestamps at which an event (entry decision) occurs.
    pt_sl : (float, float)
        (profit-take, stop-loss) thresholds in units of *fraction of price*.
        E.g. (0.02, 0.01) is +2% / -1%.
    horizon_bars : int
        Maximum number of bars to hold (vertical barrier).
    side : pd.Series, optional
        Direction of the trade per event, in {-1, +1}. If provided, the
        barriers are reflected for short events. If None, both barriers are
        applied symmetrically (and `label` is the direction of the first hit).

    Returns
    -------
    pd.DataFrame indexed by event timestamps with columns:
      - ``t1`` — timestamp of barrier hit
      - ``ret`` — realised return at ``t1`` from the entry close
      - ``label`` — +1 / -1 / 0 (profit / stop / time)
    """
    pt, sl = pt_sl
    out = []
    for t0 in events:
        if t0 not in close.index:
            continue
        i0 = close.index.get_loc(t0)
        end_i = min(i0 + horizon_bars, len(close) - 1)
        path = close.iloc[i0 + 1 : end_i + 1]
        p0 = close.iloc[i0]
        s = int(side.loc[t0]) if side is not None and t0 in side.index else 1
        rets = (path / p0 - 1.0) * s
        pt_hit = rets[rets >= pt].index.min() if (rets >= pt).any() else None
        sl_hit = rets[rets <= -sl].index.min() if (rets <= -sl).any() else None
        if pt_hit is None and sl_hit is None:
            t1 = path.index[-1] if not path.empty else t0
            ret = rets.iloc[-1] if not rets.empty else 0.0
            label = 0
        elif pt_hit is not None and (sl_hit is None or pt_hit <= sl_hit):
            t1 = pt_hit
            ret = rets.loc[pt_hit]
            label = 1
        else:
            t1 = sl_hit
            ret = rets.loc[sl_hit]
            label = -1
        out.append({"event": t0, "t1": t1, "ret": float(ret), "label": label})
    return pd.DataFrame(out).set_index("event")


def frac_diff_weights(d: float, size: int) -> np.ndarray:
    """Coefficients for fractional differencing, oldest first."""
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w[::-1])


def frac_diff(series: pd.Series, d: float, thresh: float = 1e-3) -> pd.Series:
    """Fixed-width-window fractional differencing (López de Prado).

    Picks the truncation width so that dropped (oldest) weights contribute
    less than ``thresh`` of the total absolute weight. Returns a Series of
    the same length, NaN where the window doesn't yet cover.
    """
    if d <= 0:
        return series.copy()
    n = len(series)
    # Weights ordered most-recent first: [w_0, w_1, ..., w_{n-1}].
    w_recent_first = frac_diff_weights(d, n)[::-1]
    cum = np.cumsum(np.abs(w_recent_first))
    # Smallest width whose kept weight covers >= (1 - thresh) of total.
    target = (1.0 - thresh) * cum[-1]
    width = int(np.searchsorted(cum, target)) + 1
    width = max(2, min(width, n))
    # Reverse back to oldest-first for convolution
    w = w_recent_first[:width][::-1]
    vals = series.values
    out = np.full(n, np.nan)
    for t in range(width - 1, n):
        out[t] = float((w * vals[t - width + 1 : t + 1]).sum())
    return pd.Series(out, index=series.index, name=f"fracdiff_{d}")
