# PatchTST, iTransformer, TimesNet — the transformer family for time series

The vanilla "language transformer" applied to time series, naively, is mediocre. It has too many parameters, learns positional embeddings that don't suit numerical sequences, and treats time series tokens identically to language tokens. The recent wave of *time-series-specific* transformers — PatchTST (2023), iTransformer (2024), TimesNet (2023) — fix specific deficiencies and routinely match or beat the older architectures.

This chapter is the working summary of what's actually new.

## The core problem with naive transformers on time series

A standard transformer takes a sequence of "tokens" and computes self-attention across them. For a length-$L$ time series, that's $L$ tokens, $O(L^2)$ attention. For $L = 1000$, that's a million attention scores — feasible but expensive.

The bigger problem: tokens in language carry rich semantics; tokens in time series are scalars (or short feature vectors). The transformer's expressive power is wasted on tiny tokens. The fixes group nearby time steps into bigger units before attention.

## PatchTST — patches of the time series

Nie et al. (2023). The clean idea: split the time series into **non-overlapping patches** of length $P$ (e.g., 16), embed each patch, then apply transformer attention to the patches. For a length-1024 series with $P=16$, you have 64 patches — far cheaper than 1024 tokens, and each patch carries enough information to be a meaningful token.

A schematic implementation:

```python
import torch, torch.nn as nn

class PatchTSTBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model); self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )
    def forward(self, x):
        h, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x


class PatchTST(nn.Module):
    def __init__(self, seq_len=512, patch_len=16, stride=16, n_features=1,
                 d_model=128, n_layers=4, n_heads=8, forecast_len=24):
        super().__init__()
        assert (seq_len - patch_len) % stride == 0
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.patch_embed = nn.Linear(patch_len * n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        self.blocks = nn.ModuleList([PatchTSTBlock(d_model, n_heads) for _ in range(n_layers)])
        self.head = nn.Linear(self.n_patches * d_model, forecast_len)
        self.patch_len, self.stride = patch_len, stride

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        b = x.size(0)
        # unfold into patches
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)  # (b, n_patches, n_feat, patch_len)
        patches = patches.permute(0, 1, 3, 2).reshape(b, self.n_patches, -1)
        h = self.patch_embed(patches) + self.pos
        for block in self.blocks:
            h = block(h)
        return self.head(h.flatten(1))
```

For multivariate inputs, PatchTST uses **channel independence** — apply the same model per channel and let attention work on patches *within* a channel. Surprisingly, this often beats jointly attending across channels.

## iTransformer — invert the perspective

Liu et al. (2024). Notice that the natural sequence in multivariate time series isn't time; it's **variables**. iTransformer treats each variable's whole time series as a single "token", then applies attention *across variables*. The result: the transformer learns cross-variable interactions explicitly, while a simple feed-forward block handles time.

This sounds backwards but is empirically excellent on multivariate forecasting benchmarks (electricity load, weather, exchange rates). For trading: think of each asset as a variable; iTransformer learns cross-asset dependencies.

The code is mostly standard transformer with the input axes swapped. The `Time-Series-Library` (Tsinghua) and `neuralforecast` both ship implementations.

## TimesNet — convert to 2D

Wu et al. (2023). Identify the dominant periods in the input via FFT, then reshape the 1D series into a 2D tensor (period × within-period), and apply 2D convolutions. This explicitly exploits the multi-period structure of many time series (intra-day, weekly, monthly cycles).

For financial data with strong intraday seasonality (futures, equity intraday), TimesNet's 2D framing genuinely helps. For pure-noise daily-bar return prediction, it doesn't.

## How well do they actually work on finance?

Honest assessment, based on published benchmarks and our experience:

| Task | Best architecture |
|---|---|
| Univariate daily return prediction | GBM > linear ≈ small TCN |
| Multivariate daily return panel | GBM ≈ iTransformer (close) |
| Intraday 5-min return prediction | TCN ≈ small transformer |
| LOB short-horizon prediction | Transformer / DeepLOB |
| Multi-horizon volume forecasting | N-HiTS > PatchTST |
| Vol surface prediction | GP > GBM > NN |

The transformer family doesn't dominate. It wins in specific contexts and ties or loses elsewhere. **Always benchmark against a GBM baseline first.**

## Training tips

For time-series transformers:

- **Patch length 8-32** for most settings. 16 is a sensible default.
- **AdamW** with `betas=(0.9, 0.98)`, `weight_decay=0.05`.
- **Learning rate warmup** (1000 steps) + cosine schedule.
- **Layer norm before attention/MLP** (pre-norm), not after.
- **GELU activations**, not ReLU.
- **Gradient clipping at 1.0**.
- **No positional encoding** for many time-series tasks — the patch order gives enough.

The `neuralforecast` library has battle-tested implementations of PatchTST and iTransformer:

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST

nf = NeuralForecast(
    models=[PatchTST(h=24, input_size=512, max_steps=2000,
                     patch_len=16, n_heads=8, n_layers=3)],
    freq="H",
)
nf.fit(df_panel)
```

For real work, use the library; don't hand-roll unless you're researching architecture changes.

## A worked example: PatchTST on synthetic data

```python
import torch
import torch.nn as nn
import numpy as np

# Simulated: AR(1) with noise
rng = np.random.default_rng(0)
T = 4000
x = np.zeros(T, dtype="float32")
for t in range(1, T):
    x[t] = 0.85 * x[t-1] + rng.normal(0, 0.1)

# Build windows: input 256 → forecast next 24
def make_windows(x, input_len, forecast_len, stride=1):
    X, y = [], []
    for i in range(0, len(x) - input_len - forecast_len + 1, stride):
        X.append(x[i:i+input_len])
        y.append(x[i+input_len:i+input_len+forecast_len])
    return np.stack(X)[:, :, None], np.stack(y)

X, y = make_windows(x, 256, 24, stride=4)
X_t = torch.from_numpy(X.astype("float32"))
y_t = torch.from_numpy(y.astype("float32"))

model = PatchTST(seq_len=256, patch_len=16, stride=16, n_features=1, d_model=64,
                 n_layers=2, n_heads=4, forecast_len=24)
optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

for epoch in range(50):
    pred = model(X_t)
    loss = nn.functional.mse_loss(pred, y_t)
    optim.zero_grad(); loss.backward(); optim.step()
print(f"final loss: {loss.item():.4f}")
```

For finance, the same pattern applies — but always benchmark against a GBM trained on the same forecast horizon. If GBM beats the transformer by > 5% out-of-sample, the transformer's complexity isn't justified.

## Pitfalls

!!! warning "Transformers overfit small datasets fast"
    Below ~10k training windows, a transformer with 100k+ parameters memorises. Use a TCN instead.

!!! warning "Channel-mixing vs channel-independence"
    For genuinely-shared signal across channels (cross-asset), channel-mixing (iTransformer style) wins. For independent series (one model for many separate assets), channel-independence (PatchTST) wins. Test both.

!!! warning "Patch length too long vs the relevant timescale"
    If your signal lives at the 5-bar timescale and your patch is 32 bars, the model can't see the signal. Choose patch ≤ half the typical signal scale.

!!! warning "Forecast length too long"
    Transformer forecasters tend to *underestimate* uncertainty at long horizons. For multi-step forecasts, use conformal prediction (Module 10) for honest intervals.

## Bottom line

For deep time-series in 2026:

- **PatchTST** is a strong default for multivariate forecasting with engineered features.
- **iTransformer** when you want explicit cross-variable interactions.
- **TimesNet** when the series has obvious periodic structure.
- **Use `neuralforecast`** for production work; hand-roll only for research.
- **Always benchmark against a GBM**; deep wins less often than papers suggest.

Continue to **[Mamba and state-space models](04-mamba.md)**.
