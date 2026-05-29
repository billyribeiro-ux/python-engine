# Diffusion models for synthetic market paths

Diffusion models — the technology behind Stable Diffusion, DALL-E 3, and most modern generative AI — are surprisingly applicable to financial time series. They generate realistic synthetic data: paths that match the statistical properties of historical data (volatility clustering, fat tails, correlation structure) without copying any specific path. For stress testing, scenario generation, and data augmentation, that's enormously useful.

This chapter is the working introduction.

## The mechanism

A diffusion model trains in two stages:

1. **Forward process**: starting from real data $x_0$, progressively add Gaussian noise across $T$ timesteps until you reach pure noise $x_T$.
2. **Reverse process**: train a neural network to **denoise** — given $x_t$ (noisy data at step $t$), predict either $x_0$ or the noise that was added.

At generation time, start from pure noise and apply the trained denoiser iteratively to get a sample from the data distribution.

For finance: $x_0$ is a (T-bar) historical return path. After training, you can sample fresh return paths that look statistically like training data.

## A schematic in PyTorch

```python
import torch, torch.nn as nn
import numpy as np

def linear_beta_schedule(T, start=1e-4, end=0.02):
    return torch.linspace(start, end, T)


class DiffusionModel:
    def __init__(self, net, T=1000):
        self.net = net
        self.T = T
        beta = linear_beta_schedule(T)
        self.alpha = 1.0 - beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def forward_diffuse(self, x0, t):
        """Add noise to x0 to get x_t."""
        noise = torch.randn_like(x0)
        a = self.alpha_bar[t].sqrt().view(-1, 1)
        b = (1 - self.alpha_bar[t]).sqrt().view(-1, 1)
        return a * x0 + b * noise, noise

    def loss(self, x0):
        t = torch.randint(0, self.T, (x0.size(0),))
        x_t, noise = self.forward_diffuse(x0, t)
        pred_noise = self.net(x_t, t)
        return nn.functional.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, n, dim):
        x = torch.randn(n, dim)
        for t in reversed(range(self.T)):
            t_batch = torch.full((n,), t, dtype=torch.long)
            pred_noise = self.net(x, t_batch)
            alpha_t = self.alpha[t]; alpha_bar_t = self.alpha_bar[t]
            beta_t = 1 - alpha_t
            x = (1 / alpha_t.sqrt()) * (x - beta_t / (1 - alpha_bar_t).sqrt() * pred_noise)
            if t > 0:
                x = x + beta_t.sqrt() * torch.randn_like(x)
        return x
```

The `net` is your denoiser — typically a small UNet for images, or for 1D time series a TCN / small transformer with a time embedding for the diffusion step.

## A worked example: synthetic SPY return paths

```python
import torch, torch.nn as nn
import numpy as np

# Stack historical return paths as training data
# Each "image" is a length-T return vector
SEQ_LEN = 64

class TimeEmbed(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        emb = torch.exp(-np.log(10000) * torch.arange(half) / half).to(t.device)
        emb = t.float()[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class Denoiser(nn.Module):
    def __init__(self, seq_len, hidden=128):
        super().__init__()
        self.time_embed = TimeEmbed(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(seq_len + hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, seq_len),
        )
    def forward(self, x, t):
        emb = self.time_embed(t)
        return self.mlp(torch.cat([x, emb], dim=-1))


# Training
historical_paths = torch.from_numpy(your_T_by_seq_len_returns).float()
model = DiffusionModel(net=Denoiser(SEQ_LEN), T=200)
optim = torch.optim.AdamW(model.net.parameters(), lr=1e-3)
for _ in range(500):
    idx = torch.randint(0, len(historical_paths), (64,))
    x0 = historical_paths[idx]
    loss = model.loss(x0)
    optim.zero_grad(); loss.backward(); optim.step()

# Generate synthetic paths
synthetic = model.sample(n=100, dim=SEQ_LEN)
```

After training, `synthetic` is 100 fresh paths. They won't be copies of any historical path, but their **distributional properties** (autocorrelation, volatility clustering, fat tails, etc.) will match the historical data.

## What diffusion is genuinely useful for

### 1. Stress test scenarios

Generate hundreds of crisis-like paths to test your strategy's resilience. Real crises (2008, 2020) give you exactly one sample. Diffusion gives you many.

### 2. Backtesting data augmentation

Train your strategy's ML model on both real data and diffusion-generated synthetic data. This is the most active research area; mixed results in finance (synthetic data isn't fully indistinguishable from real, and overfitting to synthetic hurts).

### 3. Backfilling missing data

Train on series where data is complete, generate conditional fills for series with gaps.

### 4. Counterfactual scenarios

With conditional diffusion, you can sample paths conditioned on "what if VIX was 50 instead of 20?" — useful for portfolio stress testing.

## What diffusion is NOT good at (in finance)

- **Forecasting** — diffusion samples from a marginal distribution, not a predictive distribution given context. For prediction, use the transformers and SSMs from previous chapters.
- **Tiny datasets** — diffusion needs lots of data to learn the manifold. <1,000 paths is not enough.
- **Online learning** — diffusion is trained in batch; not designed for streaming updates.

## Tested implementations

For production work, the standard libraries:

- [`diffusers`](https://github.com/huggingface/diffusers) (HuggingFace) — primarily image-focused but the 1D APIs are there.
- [`TSDiff`](https://github.com/amazon-science/unconditional-time-series-diffusion) — Amazon's time-series-specific diffusion.
- [`CSDI`](https://github.com/ermongroup/CSDI) — conditional diffusion for imputation of missing time series.

## Pitfalls

!!! warning "Mode collapse"
    Diffusion models in low-data regimes can collapse — generated samples look identical to a few training paths. Inspect samples' diversity (pairwise distances, manifold coverage) before trusting them.

!!! warning "Statistics that match aren't statistics that matter"
    Synthetic paths can match first/second moments while missing the tails — and the tails are what stress tests need. Always evaluate generated paths on the metrics that matter to your application (VaR, CVaR, drawdown distribution).

!!! warning "Latent dimension and architecture choice"
    Too small a denoiser → boring outputs. Too large → memorises training data. Cross-validate with a "discriminator" score (train a classifier to tell real from synthetic — accuracy near 50% is good).

!!! warning "Time costs"
    Generating 1000 paths with T=1000 diffusion steps × 64-bar paths costs about 1000 forward passes. Use DDIM (deterministic samplers) for 5-10× speedup, or distillation for further.

## Bottom line

Diffusion models are the right tool for **generating** realistic financial scenarios. They are not the right tool for *prediction*. Use them for:

- Stress testing strategies against synthetic crises.
- Data augmentation when you have rare events you want more of.
- Counterfactual scenario generation for risk management.

For all of the above, the cost of getting them right is real but justifiable for a portfolio you can't afford to misjudge.

One more architecture class rounds out the module: pretrained **time-series foundation models**, the biggest shift in the field since 2024.

Continue to **[Time-series foundation models](07-foundation-models.md)**.
