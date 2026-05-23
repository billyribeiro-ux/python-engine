# When deep learning beats gradient boosting

The short answer: rarely, on the kind of tabular tasks that dominate quant research. The long answer: in specific domains where the data is large enough, the structure is rich enough, and the loss landscape rewards expressive representations, deep nets win cleanly. This chapter is the decision framework.

## The cases where deep wins

### 1. Limit-order-book prediction

Millions of L2/L3 events per day per symbol; clear sequential structure (bid-ask updates create patterns); features are themselves time series of orders. **Deep models (TCN, LSTM, transformer) outperform GBMs at LOB short-horizon return prediction.** The leading academic papers (Sirignano-Cont 2019, Tsantekidis et al.) show 5-15% relative improvements over hand-crafted features + linear models.

### 2. Multi-asset / multi-task learning

When you want one model that predicts returns for 5,000 stocks simultaneously, deep nets shine. A transformer can learn shared representations across symbols (sector dynamics, factor exposures) in a way that's awkward to express in GBM. The OpenAI / Two Sigma / Renaissance-style "one massive model, one big training run" approach is deep.

For a single ticker's daily prediction, this advantage doesn't matter.

### 3. Alternative data with raw modalities

Sentiment from text, sentiment from news images, satellite imagery, tick-by-tick voice transcripts of earnings calls. These are exactly what deep nets were built for. Pre-trained backbones (BERT/RoBERTa for text, ResNet/ViT for images) plus a small fine-tune head usually beats hand-crafted features.

### 4. Generative modelling

Synthetic data generation for stress testing — diffusion models, variational autoencoders. GBMs don't generate; deep nets do. Chapter 6 covers this.

### 5. Continuous-time / irregular sampling

Neural ODEs and Neural SDEs handle irregularly-sampled data natively. GBMs require fixed feature vectors. For asynchronous events (tick data, news events, fundamentals releases), the continuous-time framing is cleaner.

## The cases where gradient boosting wins (most of the time)

- Daily-bar prediction on a single asset.
- Cross-sectional features over a few hundred stocks per day.
- Datasets under ~100k samples.
- When you have <50 well-engineered features.
- When you need fast retraining (monthly walk-forward).
- When you need explainable predictions.

If your problem looks like that — and most retail and small-fund quant research does — start with XGBoost and only escalate to deep when you've exhausted that.

## The economics

For an entry-level researcher: a GBM costs $10-50 in cloud compute for a full year's research. A serious deep-learning effort starts at $1,000-5,000/month for GPU time, plus weeks of engineering. The deep approach has to pay for that overhead in better OOS Sharpe, and it often doesn't.

## A useful heuristic

> If your training data is **less than ~100k rows** and your features are **already engineered**, use a GBM. If your training data is **millions of rows** or your features need to be **learned from raw modalities**, use deep.

For the gray zone (100k-1M rows, partial feature engineering), benchmark both. Don't assume.

## What's actually new in deep time series

Since the original "LSTM for stock prediction" papers circa 2015-2017:

- **Temporal Convolutional Networks (TCN)** — fully-convolutional, parallelisable, often as good as LSTMs.
- **N-BEATS / N-HiTS** — pure feed-forward forecasting nets that beat statistical baselines on M4 / M5 competitions.
- **PatchTST / iTransformer / TimesNet** — transformer variants designed for time series rather than language.
- **Mamba / S4** — state-space models that get RNN-like inference speed with transformer-like training parallelism.
- **Neural ODEs / SDEs** — continuous-time generalisations.
- **Diffusion models** — generative modeling at scale.

The rest of this module is the working introduction to each. None of them are magic for finance; all of them are useful tools in the right context.

## A specific anti-pattern to avoid

```python
# 1. Take 5 years of SPY daily bars (~1,260 rows)
# 2. Train a 12-layer transformer
# 3. Wonder why CV Sharpe is amazing and live Sharpe is zero
```

The transformer has more parameters than data. It memorised the training set. The CV score reflects the memorisation, not the (non-existent) edge.

For 1,000-row datasets, deep is the wrong tool. Use linear regression. Maybe.

## Engineering overhead

Deploying a deep model in production adds:

- **GPU inference servers** ($500-2000/month at modest scale).
- **Model versioning** (MLflow, Weights & Biases).
- **More careful CV** because of training instability across seeds.
- **More careful monitoring** because deep nets fail in less interpretable ways.

If you don't have the engineering to support all of this, the marginal Sharpe gain isn't worth it. Pick the simpler tool.

## The realistic picture

A pragmatic quant shop in 2026 looks something like:

- **80% of strategies**: gradient-boosted models with engineered features, monthly retrained, walk-forward backtested.
- **15% of strategies**: lighter deep models (TCN, small transformer) for cross-asset learning or for medium-frequency tasks.
- **5% of strategies**: serious deep nets (large transformer, diffusion for stress tests) — usually research-stage, occasionally in production at large funds.

This module covers the 15% and 5%. Don't let the glamour fool you into ignoring the 80%.

Continue to **[TCN, N-BEATS, N-HiTS](02-tcn-nbeats.md)**.
