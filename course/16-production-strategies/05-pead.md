# Post-earnings drift with bias controls

Post-Earnings-Announcement Drift (PEAD) is one of the oldest and best-documented anomalies in finance (Ball & Brown, 1968; Bernard & Thomas, 1989). The pattern: stocks that beat earnings expectations continue to outperform for 1-3 months after the announcement; stocks that miss continue to underperform.

The anomaly has been known for 50+ years and *still works* — it has decayed but not died, surviving in modified form. This chapter shows how to build a clean PEAD strategy with the bias controls that separate working alpha from look-ahead-driven noise.

## The mechanism

Three economic stories:

1. **Slow information diffusion** — analysts revise estimates over weeks, not minutes. The full impact of a beat isn't priced in immediately.
2. **Anchoring and conservatism** — investors update beliefs slower than Bayes would.
3. **Institutional friction** — index funds rebalance on calendar dates; active managers wait for "confirmation."

Whichever story you believe, the empirical pattern is consistent: SUE (Standardised Unexpected Earnings) > 0 → outperforms for ~60 days; SUE < 0 → underperforms.

## SUE — the standard signal

For a stock that just reported:

$$
\text{SUE} = \frac{\text{Actual EPS} - \text{Median Analyst Estimate}}{\text{Std of Analyst Estimates}}
$$

Or equivalently:

$$
\text{SUE} = \frac{\text{Actual} - \text{Expected}}{\text{Standard Deviation of Forecast Errors}}
$$

Either gives a unit-free measure of the surprise. SUE > 2 is a meaningful positive surprise; SUE < -2 is meaningful negative.

## Bias controls

The list of leakage traps for PEAD is long:

1. **Point-in-time earnings.** Use the *original* announcement, not the restated value. Many databases give you the restated number, which is what the analyst saw months later — not what was published.
2. **Point-in-time estimates.** Use the consensus estimate that existed *before* the announcement, not the post-announcement consensus.
3. **Survivorship.** Use the historical S&P 500 (or your universe), not today's.
4. **Calendar bias.** PEAD effects cluster around earnings season; if your backtest is biased toward those quarters' returns, the alpha is fictitious.
5. **Look-ahead in the "expected" number.** If you used "actual EPS plus revision noise" as the expected, you've leaked the answer.
6. **Survivorship in the analyst data.** Brokers that went out of business have their reports removed from databases.

For a research project, get a vendor like FactSet, Refinitiv, or Wharton's IBES with proper point-in-time semantics.

## The strategy

```python
import pandas as pd
import numpy as np


def pead_signal(sue: pd.Series, threshold: float = 1.5) -> pd.Series:
    """+1 if SUE > threshold, -1 if SUE < -threshold, 0 otherwise."""
    sig = pd.Series(0.0, index=sue.index)
    sig[sue > threshold] = +1
    sig[sue < -threshold] = -1
    return sig


def pead_strategy(earnings_events: pd.DataFrame, returns_panel: pd.DataFrame,
                   hold_days: int = 60) -> pd.Series:
    """
    earnings_events: DataFrame with columns [symbol, announcement_date, sue].
    returns_panel: T x N daily returns DataFrame, index=date, columns=symbols.
    Holds each position for `hold_days` after announcement.
    """
    out = pd.DataFrame(0.0, index=returns_panel.index, columns=returns_panel.columns)
    for _, row in earnings_events.iterrows():
        sym = row["symbol"]
        ann = pd.Timestamp(row["announcement_date"])
        sue = row["sue"]
        if sym not in out.columns:
            continue
        sig = +1 if sue > 1.5 else -1 if sue < -1.5 else 0
        if sig == 0:
            continue
        # Trade from announcement+1 to announcement+hold_days (no same-day position)
        mask = (out.index > ann) & (out.index <= ann + pd.Timedelta(days=hold_days))
        out.loc[mask, sym] += sig
    return out


def pead_portfolio(signal_panel: pd.DataFrame, returns_panel: pd.DataFrame,
                    target_vol: float = 0.10) -> pd.Series:
    """Equal vol-weight across all active PEAD positions."""
    vol = returns_panel.rolling(60).std() * np.sqrt(252)
    sized = signal_panel / vol.replace(0, np.nan)
    gross = sized.abs().sum(axis=1)
    weights = sized.div(gross.replace(0, np.nan), axis=0)
    pnl_raw = (weights.shift(1) * returns_panel).sum(axis=1)
    realised = pnl_raw.rolling(60).std() * np.sqrt(252)
    scale = (target_vol / realised.replace(0, np.nan)).clip(upper=3.0).shift(1).fillna(1.0)
    return pnl_raw * scale
```

The `out.index > ann` is critical — entering the day after announcement (not same-day) avoids the noisy first-day jump and rules out same-bar leakage.

## Walk-forward validation

Re-test the SUE threshold and hold days annually, walk-forward. Common pattern:

- **Threshold**: 1.0, 1.5, 2.0 — try each, take the most consistent.
- **Hold days**: 30, 60, 90. Effect typically peaks around 45-60 days.
- **Universe**: large-cap only (PEAD on micro-caps is dominated by liquidity friction).

## Realistic costs

For a strategy holding 60 days per name with 10-20 positions at a time, turnover is ~6-8x annually. At 10 bps round-trip on liquid mid-to-large caps, that's 60-80 bps of cost per year.

If gross alpha is 4%, net is 3.2%. Still meaningful, but the cost is a third of the gross.

## Modern variants

Vanilla PEAD has decayed (from ~6% gross in the 1990s to ~3% in the 2020s). Modern variants:

- **Conditional on revisions** — only trade when both SUE AND post-announcement analyst revisions are positive (negative).
- **Conditional on guidance** — trade only when management guidance reinforces the SUE direction.
- **Cross-sectional**: long top-decile SUE, short bottom-decile, rebalance monthly. Dollar-neutral; lower idiosyncratic risk.

The cross-sectional version typically delivers Sharpe ~0.6-0.8 net, with very low correlation to broad market beta.

## A worked example

You'd need real earnings data to backtest seriously. The structure (without the data layer):

```python
# Pretend you have:
# earnings_df = pd.DataFrame({
#     "symbol": [...],
#     "announcement_date": [...],
#     "actual_eps": [...],
#     "consensus_eps": [...],  # pre-announcement consensus
#     "consensus_std": [...],  # std of analyst forecasts
# })

# earnings_df["sue"] = (earnings_df["actual_eps"] - earnings_df["consensus_eps"]) \
#                      / earnings_df["consensus_std"]

# returns_panel = ... # daily returns for the universe of symbols

signal_panel = pead_strategy(earnings_df, returns_panel, hold_days=60)
pnl = pead_portfolio(signal_panel, returns_panel, target_vol=0.10)
print(f"Net Sharpe (gross): {pnl.mean() / pnl.std() * np.sqrt(252):.2f}")
```

Module 5's data engineering covers how to set up point-in-time earnings storage that supports this kind of backtest correctly.

## Risk overlays

- **No PEAD positions in stocks under $5/share** — these have wide spreads and big idiosyncratic moves.
- **Skip if implied move > 20%** — earnings was already a huge event; drift is dominated by other factors.
- **Position-size cap**: no single PEAD position > 5% of total gross.

## Pitfalls

!!! warning "Data quality is the entire ballgame"
    With sloppy SUE numbers, your "backtest Sharpe" is fiction. With clean point-in-time data, the alpha is real but modest.

!!! warning "Decay over time"
    PEAD has decayed dramatically since publication. Today's strategy has half the historical gross alpha. Don't backtest on 1980s data and assume it's tradeable today.

!!! warning "Crowding"
    Many funds run PEAD variants. After a big surprise, the price gaps faster now than in 1985 — much of the drift is captured in minutes, not weeks.

!!! warning "Earnings revisions vs SUE"
    Modern research suggests revisions (changes in consensus *after* announcement) are a better predictor than the pre-announcement SUE. Worth including in the signal.

## Bottom line

A production PEAD strategy:

- **Point-in-time consensus + actual EPS**, SUE thresholding.
- **Hold 60 days after announcement** (not same-day).
- **Cross-sectional long-short** for lower idiosyncratic risk.
- **Universe filter**: liquid large-cap, no penny stocks.
- **Position-size caps + earnings-clustering checks**.

Net Sharpe: ~0.5-0.7 in production. Lower correlation to market = a useful diversifier.

Continue to **[Calendar and diagonal spreads on IV term-structure](06-calendar-spreads.md)**.
