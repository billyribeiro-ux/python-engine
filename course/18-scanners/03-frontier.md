# Frontier scanners

These scanners exploit ideas from Module 17 — Hawkes processes, persistent homology, transfer entropy, SVI residuals, dealer GEX. Few retail traders run them; few funds have all of them. The information they provide is genuinely complementary to traditional signals.

## 1. Dealer GEX inflection scanner

The setup: net dealer gamma exposure (Module 15 chapter 5) is positive but the **zero-gamma level** is near current spot. A small downward move can flip the regime from "dampening" to "amplifying" — historically associated with sudden vol spikes.

```python
import pandas as pd
import numpy as np
from engine.options import greeks
from engine.scanners import scanner


def gex_at_spot(chain: pd.DataFrame, spot: float, r=0.045, q=0.013) -> float:
    """Net dealer GEX (long puts - short calls) in $ per 1% spot move."""
    total = 0.0
    for _, row in chain.iterrows():
        T = max((row["expiry"] - pd.Timestamp.utcnow()).days, 1) / 365
        g = greeks(spot, row["strike"], T, r, q, row["implied_volatility"],
                    kind="call" if row["right"] == "C" else "put")
        sign = +1 if row["right"] == "P" else -1     # dealer long puts, short calls
        total += sign * g.gamma * row["open_interest"] * 100 * spot ** 2 * 0.01
    return float(total)


def find_zero_gamma(chain: pd.DataFrame, spot: float, range_pct: float = 0.05):
    """Spot level where net GEX crosses zero, within ±range_pct."""
    from scipy.optimize import brentq
    lo, hi = spot * (1 - range_pct), spot * (1 + range_pct)
    f = lambda s: gex_at_spot(chain, s)
    try:
        return float(brentq(f, lo, hi))
    except ValueError:
        return float("nan")


@scanner("gex-inflection")
class GEXInflectionScanner:
    """Flag underlyings where net GEX zero-crossing is near current spot."""
    def __init__(self, max_distance_pct: float = 0.02):
        self.max_distance_pct = max_distance_pct

    def scan(self, universe, asof):
        rows = []
        for sym in universe:
            try:
                spot = fetch_spot(sym, asof)
                chain = fetch_chain(sym, asof)
                current_gex = gex_at_spot(chain, spot)
                zero_g = find_zero_gamma(chain, spot)
                if not np.isfinite(zero_g):
                    continue
                distance_pct = (zero_g - spot) / spot
                if abs(distance_pct) < self.max_distance_pct:
                    rows.append({"symbol": sym, "spot": spot, "current_gex": current_gex,
                                  "zero_gamma": zero_g, "distance_pct": distance_pct})
            except Exception:
                continue
        return pd.DataFrame(rows).sort_values("distance_pct", key=abs)
```

Output: a list of underlyings where one or two days of price action could flip the dealer-positioning regime. These are fragile setups worth watching.

## 2. Vol-smile dislocation (SVI residuals)

The setup: fit an SVI smile (Module 15 chapter 1) to the observed IVs; flag the strike-expiry pairs where the market IV is far from the fitted smile.

```python
@scanner("svi-dislocation")
class SVIDislocationScanner:
    def __init__(self, residual_threshold: float = 2.0):
        # 2σ residual in IV-volatility units
        self.residual_threshold = residual_threshold

    def scan(self, universe, asof):
        rows = []
        for sym in universe:
            try:
                chain = fetch_chain(sym, asof)
                # Per-expiry SVI fit
                for expiry, group in chain.groupby("expiry"):
                    if len(group) < 5: continue
                    spot = fetch_spot(sym, asof)
                    log_m = np.log(group["strike"] / spot)
                    T = (expiry - asof).days / 365
                    w_obs = group["implied_volatility"] ** 2 * T
                    fit = fit_svi_slice(log_m.values, w_obs.values)
                    w_fit = svi_total_variance(log_m.values, **fit)
                    iv_fit = np.sqrt(np.maximum(w_fit / T, 0))
                    residuals = (group["implied_volatility"] - iv_fit) / group["implied_volatility"].std()
                    for i, r in enumerate(residuals):
                        if abs(r) > self.residual_threshold:
                            rows.append({"symbol": sym, "expiry": expiry,
                                          "strike": group["strike"].iloc[i],
                                          "right": group["right"].iloc[i],
                                          "iv": group["implied_volatility"].iloc[i],
                                          "iv_fit": iv_fit[i], "residual_z": r})
            except Exception:
                continue
        return pd.DataFrame(rows).sort_values("residual_z", key=abs, ascending=False)
```

A point that lives 2-3σ above the smile is candidate-rich (option overpriced); 2-3σ below is candidate-cheap (underpriced). These are tradeable by selling the rich, buying the cheap with a delta hedge.

## 3. Hawkes intensity spike

The setup: fit a Hawkes process (Module 17 chapter 1) to recent news/event timestamps for a symbol; flag when current intensity is multiples above the baseline.

```python
@scanner("hawkes-spike")
class HawkesSpikeScanner:
    def __init__(self, intensity_ratio_threshold: float = 3.0):
        self.ratio_threshold = intensity_ratio_threshold

    def scan(self, universe, asof):
        rows = []
        for sym in universe:
            recent_events = fetch_recent_news_events(sym, asof - pd.Timedelta(days=30), asof)
            if len(recent_events) < 10: continue
            params = fit_hawkes_params(sym)        # pre-fitted on long history
            current_intensity = hawkes_intensity_now(recent_events, asof.timestamp(), **params)
            baseline = params["mu"]
            ratio = current_intensity / baseline if baseline > 0 else float("inf")
            if ratio > self.ratio_threshold:
                rows.append({"symbol": sym, "current_intensity": current_intensity,
                              "baseline": baseline, "ratio": ratio,
                              "recent_event_count": len(recent_events)})
        return pd.DataFrame(rows).sort_values("ratio", ascending=False)
```

A 5-10× spike vs baseline indicates a self-exciting event cluster — short-term realised vol is likely elevated. Use as a sizing input or a momentum-strategy entry filter.

## 4. TDA fragility scanner

The setup: compute persistent homology summaries (Module 17 chapter 2) of rolling correlation matrices in the universe; flag when fragility metrics decline sharply.

```python
@scanner("tda-fragility")
class TDAFragilityScanner:
    """Flag when the cross-sectional correlation network's topological structure
    collapses (i.e., everything becoming correlated)."""
    def __init__(self, drop_threshold: float = 0.5):
        self.drop_threshold = drop_threshold

    def scan(self, universe, asof):
        # universe: sector ETFs or broad cross-section
        returns_panel = fetch_returns_panel(universe, asof - pd.Timedelta(days=120), asof)
        # H1 lifetime today vs 20 days ago
        today_corr = returns_panel.tail(60).corr().values
        baseline_corr = returns_panel.iloc[-80:-20].corr().values
        try:
            today_pd = correlation_persistence(today_corr, max_dim=1)
            base_pd = correlation_persistence(baseline_corr, max_dim=1)
            today_summary = persistence_summary(today_pd)
            base_summary = persistence_summary(base_pd)
            ratio = today_summary["total_h1_lifetime"] / (base_summary["total_h1_lifetime"] + 1e-6)
            if ratio < self.drop_threshold:
                return pd.DataFrame([{
                    "scope": "universe", "today_h1_lifetime": today_summary["total_h1_lifetime"],
                    "baseline_h1_lifetime": base_summary["total_h1_lifetime"], "ratio": ratio,
                }])
        except Exception:
            pass
        return pd.DataFrame()
```

A single-row output: when the H1 lifetime drops to under 50% of baseline, the asset network has lost its sub-cluster structure → correlation regime has shifted → tighten risk overlays.

## 5. Transfer-entropy lead-lag scanner

The setup: maintain a lead-lag network from TE (Module 17 chapter 3); flag pairs where the leader has just moved sharply.

```python
@scanner("te-leadlag")
class TELeadLagScanner:
    """Flag follower symbols whose 'leaders' just moved sharply."""
    def __init__(self, trigger_z: float = 2.0):
        self.trigger_z = trigger_z

    def scan(self, universe, asof):
        rows = []
        # Pre-fitted lead-lag network (refit weekly)
        network = load_te_network()
        recent_returns = fetch_recent_returns(universe, asof, lookback=60)
        zscores = (recent_returns.iloc[-1] - recent_returns.mean()) / recent_returns.std()
        for x, y, attr in network.edges(data=True):
            if x not in zscores or y not in universe: continue
            if abs(zscores[x]) > self.trigger_z:
                rows.append({"leader": x, "follower": y, "te_weight": attr.get("weight", 0.0),
                              "leader_z": float(zscores[x]),
                              "implied_direction": int(np.sign(zscores[x]))})
        return pd.DataFrame(rows).sort_values("te_weight", ascending=False)
```

Trade the follower in the direction of the leader's move. The TE-weighted ranking prioritises the most-confident edges.

## 6. Composing frontier signals

The richest signals come from **confluence**:

- A symbol flagged by GEX-inflection AND vol-smile dislocation → fragile and mispriced.
- A symbol flagged by Hawkes spike AND TE leader is moving → high near-term realised vol with a directional bias.

```python
def confluence_report(universe, asof, sources=("gex-inflection", "svi-dislocation",
                                                  "hawkes-spike", "te-leadlag")):
    from engine.scanners import registry
    from collections import Counter
    flag_counts = Counter()
    sources_per_symbol = {}
    for src in sources:
        df = registry.get(src)().scan(universe, asof)
        if "symbol" in df.columns:
            for sym in df["symbol"].unique():
                flag_counts[sym] += 1
                sources_per_symbol.setdefault(sym, []).append(src)
    return pd.DataFrame([
        {"symbol": sym, "flag_count": count, "sources": ", ".join(sources_per_symbol[sym])}
        for sym, count in flag_counts.most_common()
    ])
```

Symbols flagged by 3+ frontier scanners simultaneously are the highest-conviction watchlist additions.

## Pitfalls

!!! warning "Compute cost"
    Frontier scanners are expensive. TDA per symbol per day is minutes; Hawkes fitting is comparable. Refit infrequently (weekly for stable params; daily only for the live computations).

!!! warning "Data requirements"
    Many frontier scanners require **historical chains with greeks** or **labelled news events** that retail data feeds don't provide. Without proper data, the scanners are decorative.

!!! warning "Threshold robustness"
    Frontier signals are noisier than production ones. Use wider thresholds and tolerate fewer-but-cleaner alerts.

!!! warning "Backtest validity**
    Backtesting these scanners on truly historical data is hard because some require option-chain or news-feed history with daily granularity. Be cautious about claimed historical Sharpes.

## Bottom line

Frontier scanners give you signals **complementary to production scanners**. They're not standalone alpha sources; they're additions to a multi-layer detection system. Symbols flagged by both production AND frontier scanners get the highest research-attention budget.

Continue to **[Microstructure scanners](04-microstructure.md)**.
