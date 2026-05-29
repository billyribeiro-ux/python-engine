# Time-series foundation models

The biggest shift in time-series ML since this module's other chapters were written: **pretrained foundation models for forecasting**. Just as a language model is pretrained on a giant text corpus and then used zero-shot, a time-series foundation model is pretrained on billions of time-series observations and then forecasts a *new* series it has never seen — with no training on your data at all.

By 2026 these are mainstream tools. This chapter is the honest working guide: what they are, when they help, and why they don't repeal the low-SNR ceiling from Module 10.

## The landscape

| Model | Who | Shape | Access |
|---|---|---|---|
| **TimesFM** | Google | decoder-only, patched | open weights (Hugging Face) |
| **Chronos / Chronos-Bolt** | Amazon | tokenise values → T5; Bolt is the fast variant | open weights |
| **Moirai / Moirai-MoE** | Salesforce | masked encoder, any-variate | open weights |
| **TimeGPT** | Nixtla | proprietary | commercial API (`nixtla` pkg) |
| **Lag-Llama** | open collab | probabilistic, decoder | open weights |
| **TabPFN-TS** | Prior Labs | in-context, tabular-style | open weights |

All share the premise: pretrain once on a massive, diverse corpus of time series; then forecast any new series **zero-shot** (no fine-tuning) or **few-shot** (light fine-tuning).

## Zero-shot forecasting in practice

Chronos via the `chronos-forecasting` package:

```python
import torch
from chronos import ChronosPipeline
import numpy as np

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-base",
    device_map="cpu",                          # or "cuda"
    torch_dtype=torch.bfloat16,
)

# A context series (e.g., 200 days of something)
context = torch.tensor(history_values, dtype=torch.float32)

# Probabilistic forecast: 64 sample paths, 30 steps ahead
forecast = pipeline.predict(context, prediction_length=30, num_samples=64)
# forecast shape: (1, 64, 30)  — quantiles available by reducing over samples
median = np.quantile(forecast[0].numpy(), 0.5, axis=0)
lo, hi = np.quantile(forecast[0].numpy(), [0.1, 0.9], axis=0)
```

No training step. You hand it history; it returns a distribution of futures. TimesFM and Moirai have similar APIs.

TimeGPT (API, no local model):

```python
from nixtla import NixtlaClient

client = NixtlaClient(api_key="...")
forecast = client.forecast(df=your_dataframe, h=30, freq="D",
                            level=[80, 95])      # prediction intervals
```

## Where they genuinely help

1. **Cold start.** A new symbol, product, or sensor with little history. A foundation model forecasts from day one; a trained model needs a training set.
2. **Many series, no per-series tuning.** Forecasting volume / demand / load across thousands of series where building a bespoke model each is impractical. One foundation model covers all.
3. **Exogenous-poor series with strong seasonality.** Volume profiles, intraday seasonality, macro series — the pretraining corpus has seen a lot of these patterns.
4. **A strong baseline / feature.** The foundation model's point forecast (or its quantile spread) becomes a *feature* in your downstream GBM, often a good one.

## Where they don't help (the low-SNR ceiling holds)

This is the honest part. A foundation model **does not beat the information-theoretic ceiling** on daily equity *return* prediction. Returns are ~IID-with-fat-tails at the daily horizon; there is very little autoregressive signal, and no amount of pretraining manufactures signal that isn't there.

Empirically as of 2026:

- **Daily return direction**: foundation models ≈ random ≈ a tuned GBM on engineered features. The SNR ceiling from Module 10 chapter 1 still binds.
- **Volume / volatility / seasonality forecasting**: foundation models are genuinely competitive and often beat hand-built baselines — these series have real autoregressive structure.
- **Price *level* forecasting**: foundation models extrapolate the recent trend, which is exactly what you'd expect and rarely tradeable.

The rule from Module 12 chapter 1 stands: **the bottleneck is signal, not model capacity.** A foundation model is another way to spend capacity; it doesn't create edge in a low-SNR target.

## A pragmatic use: foundation-model features

The defensible way to use these in a trading pipeline is as a feature generator, not a standalone predictor:

```python
import numpy as np
import pandas as pd


def foundation_features(close: pd.Series, pipeline, horizon: int = 5) -> pd.DataFrame:
    """Roll a foundation model over history; emit its forecast spread as features.
    Causal: at bar t, only history up to t is used."""
    import torch
    feats = pd.DataFrame(index=close.index, columns=["fm_median", "fm_spread"], dtype=float)
    window = 200
    for t in range(window, len(close)):
        ctx = torch.tensor(close.iloc[t - window:t].values, dtype=torch.float32)
        fc = pipeline.predict(ctx, prediction_length=horizon, num_samples=32)[0].numpy()
        feats.iloc[t, 0] = np.quantile(fc[:, -1], 0.5)            # median terminal forecast
        feats.iloc[t, 1] = np.quantile(fc[:, -1], 0.9) - np.quantile(fc[:, -1], 0.1)  # uncertainty
    return feats
```

The `fm_spread` (model's own uncertainty) is often the more useful column — a regime-aware uncertainty estimate you can feed into conformal sizing (Module 10 chapter 5). The point forecast on returns rarely adds alpha; the *uncertainty* sometimes does.

!!! warning "This is slow"
    A foundation model inference per bar over years of history is expensive. Cache aggressively, batch the windows, or compute the feature on a coarser grid. For research, fine; for a tight backtest loop, precompute.

## Fine-tuning vs zero-shot

- **Zero-shot**: no training; instant; the safe default for exploration.
- **Few-shot / fine-tune**: light training on your domain. Helps when your series has structure the pretraining corpus underweighted (e.g., crypto microstructure, a specific exchange's session shape). Chronos and TimesFM both support fine-tuning.

For finance, fine-tuning on returns rarely helps (no signal to fit). Fine-tuning on volume / vol / order-flow series can.

## Evaluation discipline carries over

Everything from Module 9 applies:

- **Walk-forward, no leakage.** A foundation model's "zero-shot" still leaks if you feed it future context. Respect the causal boundary.
- **Honest baselines.** Compare against a naive forecast (last value / seasonal naive) and a tuned GBM. Foundation models that "look great" often only beat a strawman baseline.
- **Backtest with costs.** A marginally-better forecast that turns over more isn't better net.

## Pitfalls

!!! warning "Treating a great benchmark score as tradeable edge"
    Foundation models top the M-competition-style benchmarks (demand, energy, weather). Those series have real signal. Financial returns are not those series. Benchmark wins don't transfer to alpha.

!!! warning "Look-ahead via the pretraining corpus"
    If a model was pretrained on data that overlaps your backtest period (including your own assets), its "zero-shot" forecast has implicitly seen the future. For rigorous backtests, prefer models with a documented pretraining cutoff before your test window.

!!! warning "Probabilistic ≠ calibrated"
    The quantile spread a foundation model returns is *not* guaranteed calibrated on your data. Wrap it in conformal prediction (Module 10 chapter 5) if you size against the intervals.

!!! warning "Inference cost in production"
    A transformer forecast per symbol per bar is far heavier than a GBM. For a 5,000-symbol intraday universe, the compute bill is real. Budget for it or use the lighter variants (Chronos-Bolt, TimesFM small).

## Bottom line

Time-series foundation models (TimesFM, Chronos, Moirai, TimeGPT) are a real 2024–2026 development and a genuine new tool:

- **Use them** for cold-start, many-series, and seasonality/volume/vol forecasting.
- **Use their uncertainty** as a feature into conformal sizing.
- **Do not expect** them to manufacture edge in low-SNR return prediction — the ceiling from Module 10 still binds.
- **Evaluate** with the same walk-forward, honest-baseline, cost-aware discipline as everything else.

## End of Module 12

You now have the working tour of modern deep learning for time series — lightweight (TCN, N-BEATS), cutting-edge (Mamba, Neural SDEs, diffusion), and the foundation-model wave. The next module — **Reinforcement Learning** — gives you the framework for problems where the *action* matters: execution, sizing, market-making.

Continue to **[Module 13 — Reinforcement Learning](../13-reinforcement-learning/index.md)**.
