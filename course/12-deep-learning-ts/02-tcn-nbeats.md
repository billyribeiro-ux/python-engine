# TCN, N-BEATS, N-HiTS

Three families of "non-transformer" deep models for time series. All three are conceptually simpler than transformers, often as accurate, and dramatically cheaper to train. For most "predict the next step or two" tasks, one of these is the right starting point in deep land.

## Temporal Convolutional Networks (TCN)

A TCN is a stack of **dilated causal 1D convolutions**. Each convolution operates only on past inputs (causal); dilations let the receptive field grow exponentially with depth (so few layers cover a long history).

The key properties:

- **Causal by construction.** No leakage; no need for masking.
- **Parallelisable training.** Unlike RNNs.
- **Long receptive field with few parameters.** A 6-layer TCN with dilations 1, 2, 4, 8, 16, 32 covers 63 past time steps.

A working PyTorch implementation:

```python
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)
    def forward(self, x):
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(ch, ch, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.dropout(self.act(self.conv2(h)))
        return x + h                              # residual


class TCN(nn.Module):
    def __init__(self, n_features, hidden=64, n_blocks=6, kernel=3):
        super().__init__()
        self.embed = nn.Conv1d(n_features, hidden, 1)
        self.blocks = nn.ModuleList([
            TCNBlock(hidden, kernel, dilation=2 ** i) for i in range(n_blocks)
        ])
        self.head = nn.Conv1d(hidden, 1, 1)
    def forward(self, x):
        # x: (batch, n_features, seq_len)
        h = self.embed(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h).squeeze(1)            # (batch, seq_len)
```

For a daily-bar prediction task with 8 features and a 1-step-ahead target, the loss is just MSE on the last output:

```python
loss = nn.functional.mse_loss(model(x_batch)[:, -1], y_batch)
```

TCNs are the easiest deep models to get right for time series. Start here before transformers.

## N-BEATS — the pure feed-forward forecasting model

Oreshkin et al. (2020). A fully feed-forward model that won the M4 forecasting competition. Architecture: a stack of "blocks", each producing both a **backcast** (its reconstruction of the input) and a **forecast** (its prediction of the future). The backcasts are subtracted from the input as the residual flows through subsequent blocks.

The killer insight: pure feed-forward with carefully-designed residual structure can match or beat RNN/transformer forecasters at a fraction of the compute.

A minimal version:

```python
class NBeatsBlock(nn.Module):
    def __init__(self, backcast_len, forecast_len, n_neurons=256, n_layers=4):
        super().__init__()
        layers = []
        for i in range(n_layers):
            inp = backcast_len if i == 0 else n_neurons
            layers += [nn.Linear(inp, n_neurons), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)
        self.theta_back = nn.Linear(n_neurons, backcast_len)
        self.theta_fore = nn.Linear(n_neurons, forecast_len)
    def forward(self, x):
        h = self.backbone(x)
        return self.theta_back(h), self.theta_fore(h)


class NBeats(nn.Module):
    def __init__(self, backcast_len, forecast_len, n_blocks=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(backcast_len, forecast_len) for _ in range(n_blocks)
        ])
    def forward(self, x):
        # x: (batch, backcast_len)
        residual = x
        forecast_sum = 0
        for b in self.blocks:
            back, fore = b(residual)
            residual = residual - back
            forecast_sum = forecast_sum + fore
        return forecast_sum
```

For univariate forecasting (single series, no exogenous features), N-BEATS is excellent. The recent multivariate variant (NBEATSx) supports exogenous inputs.

## N-HiTS — hierarchical N-BEATS

Challu et al. (2023). N-BEATS with **multi-rate hierarchies**: each block looks at the input at a different timescale (downsampling) and forecasts a smoothed version of the target. The final forecast is the sum across hierarchies.

Why it matters: stock returns have structure at multiple scales (intraday noise, daily trends, weekly cycles). N-HiTS captures this with explicit multi-rate convolutions — often beats plain N-BEATS by 10-30% on long-horizon forecasts.

The clean implementation is in the [`neuralforecast`](https://github.com/Nixtla/neuralforecast) library:

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

nf = NeuralForecast(
    models=[NHITS(h=5, input_size=60, max_steps=500)],
    freq="B",
)
nf.fit(df_panel)              # panel DataFrame with columns unique_id, ds, y
forecasts = nf.predict()
```

The `neuralforecast` library wraps N-BEATS, N-HiTS, TFT, PatchTST and others under a uniform sklearn-like API. For practical forecasting projects, start with it.

## When TCN / N-BEATS / N-HiTS win

- **TCN** for predicting "value at next bar" given a long history of features. Easy to train, easy to interpret receptive field, parallelises well.
- **N-BEATS** for univariate forecasting with no exogenous features and a need for very fast inference.
- **N-HiTS** for long-horizon multi-frequency forecasting on liquid markets.

For trading, the typical use case is **short-horizon prediction with engineered features** — that's a TCN sweet spot. For multi-horizon volume forecasting (e.g., for VWAP execution), N-HiTS is excellent.

## A worked example: training a small TCN on synthetic data

```python
import torch
import torch.nn as nn
import numpy as np

rng = np.random.default_rng(0)
T, F = 5000, 8
X = rng.normal(0, 1, (T, F)).astype("float32")
# Synthetic target with mild autocorrelation
y = (0.3 * np.roll(X[:, 0], 1) + 0.2 * X[:, 1] - 0.1 * X[:, 2] +
     0.05 * rng.normal(0, 1, T)).astype("float32")

# Reshape for TCN: (batch=1, channels=F, seq_len=T)
x_t = torch.from_numpy(X.T).unsqueeze(0)
y_t = torch.from_numpy(y).unsqueeze(0)

model = TCN(n_features=F, hidden=32, n_blocks=4)
optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(50):
    model.train()
    pred = model(x_t)
    loss = nn.functional.mse_loss(pred, y_t)
    optim.zero_grad(); loss.backward(); optim.step()
print(f"final loss: {loss.item():.4f}")
```

A few minutes on CPU. For real work, batch the inputs, hold out a validation slice, use early stopping.

## Pitfalls

!!! warning "Look-ahead in non-causal layers"
    If you mistakenly use `padding="same"` on a 1D conv, future tokens leak into past ones. Always use `CausalConv1d` (left-pad only) for time-series predictions.

!!! warning "Tiny datasets, big models"
    A TCN with 100k parameters fit to 1,000 daily bars overfits catastrophically. Scale model size to data size.

!!! warning "Forgetting to walk-forward retrain"
    A deep model trained on 2014-2022 will eventually drift on 2024-2025. Retrain at least quarterly.

!!! warning "Optimisation matters more than architecture"
    AdamW with weight decay 1e-4, warmup + cosine LR, gradient clipping at 1.0 — these are non-negotiable for stable training. Skipping them is the #1 reason deep models "don't work" for newcomers.

## Bottom line

For deep-learning time-series forecasting:

- **TCN** as the default starting point — simple, fast, no transformer complexity.
- **N-BEATS / N-HiTS** for forecasting tasks; use the `neuralforecast` library.
- Move to transformers (next chapter) only if these aren't enough.

Continue to **[PatchTST, iTransformer, TimesNet](03-transformers-ts.md)**.
