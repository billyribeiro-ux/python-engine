# Mamba and state-space models

State-space models (SSMs) are an old idea — the Kalman filter is one — but their resurgence as a serious alternative to transformers happened in 2022-2024 with the **S4** and **Mamba** architectures. They offer transformer-level accuracy at **linear-time complexity** in sequence length, instead of quadratic. For long sequences, that's a transformative property.

This chapter is the conceptual introduction and a working sketch.

## The key trade-off, in one sentence

**Transformers** have great training parallelism (matrix multiplies everywhere) but quadratic inference cost in sequence length and limited extrapolation. **RNNs** have linear inference and unbounded receptive field but bad training parallelism. **Modern SSMs** (S4, Mamba) get both: parallel training via FFT/convolution tricks AND linear inference.

For sequences > 1024, that matters. For sequences < 256, vanilla transformers are fine.

## The continuous-time SSM

The underlying model is a linear ODE:

$$
h'(t) = A h(t) + B u(t)
$$
$$
y(t) = C h(t) + D u(t)
$$

where $u(t)$ is input, $h(t)$ the hidden state, $y(t)$ the output. After discretisation, it becomes a recurrence:

$$
h_t = \bar A h_{t-1} + \bar B u_t
$$
$$
y_t = C h_t + D u_t
$$

The recurrence is sequential at inference — but with carefully-chosen $A$ (HiPPO matrices in S4), the equivalent **convolutional** form can be computed in parallel during training. That's the trick.

## S4 — Structured State Space

Gu et al. (2022). The HiPPO matrices give the SSM strong long-context properties — it can remember thousands of tokens back. S4 sets state-of-the-art on the Long Range Arena benchmark (sequences up to 16,384 tokens).

For finance, S4-style SSMs are interesting for:

- Tick-level prediction across hundreds of thousands of events.
- Long-history dependence (full quarterly earnings season as one sequence).
- Multi-day, high-frequency strategies.

## Mamba — selective SSM

Gu and Dao (2024). Plain SSMs have *time-invariant* parameters — the recurrence doesn't change based on input. Mamba makes the SSM **input-dependent** by computing the state-transition matrix as a function of the current token. This recovers the "selective attention" property of transformers (focus on the important tokens, ignore the rest) within a linear-time model.

A schematic:

```python
import torch, torch.nn as nn

class MambaBlock(nn.Module):
    """Highly simplified Mamba sketch — see official repo for production code."""
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner,
                                kernel_size=4, groups=self.d_inner, padding=3)

        # SSM parameters
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        b, L, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[..., :L]
        x = x.transpose(1, 2)
        x = nn.functional.silu(x)

        # SSM: simplified — the real Mamba uses a CUDA scan.
        dt = nn.functional.softplus(self.dt_proj(x))
        B = self.B_proj(x); C = self.C_proj(x)
        # Discretize
        dA = torch.einsum("bld,dn->bldn", dt, -torch.abs(self.A))
        dB = torch.einsum("bld,bln->bldn", dt, B)
        # Recurrence (toy: a Python loop; real Mamba uses a parallel scan)
        h = torch.zeros(b, self.d_inner, self.d_state, device=x.device)
        ys = []
        for t in range(L):
            h = h * torch.exp(dA[:, t]) + dB[:, t] * x[:, t].unsqueeze(-1)
            y = (h * C[:, t].unsqueeze(1)).sum(-1)
            ys.append(y)
        y = torch.stack(ys, dim=1) + x * self.D
        y = y * nn.functional.silu(z)
        return self.out_proj(y)
```

This sketch doesn't match the real Mamba's CUDA-kernel speed — the production code uses a parallel scan that's roughly as fast as transformer attention. The official [`mamba-ssm`](https://github.com/state-spaces/mamba) package is what to use in practice.

## When SSMs win for finance

- **Long sequences** (> 1000 tokens). Vanilla transformers are O(L²); SSMs are O(L).
- **Tick-level modelling** at full session length (390 minutes × 60 seconds × hundreds of events).
- **Multi-day continuous models** without arbitrary truncation.

For everything shorter, plain transformers or even GBMs are competitive.

## Production status

As of 2025/2026:

- Mamba and Mamba-2 are in active research use; emerging in production at large model labs.
- For finance applications, SSMs are still **research-stage** — few teams have published convincing results yet.
- The libraries are evolving rapidly; pin versions if you're using them in production.

The bet: if you're building deep models on truly long financial sequences, SSMs are worth tracking. For mainstream daily-bar work, you can safely skip them and revisit in a year.

## A trap: SSMs and "infinite context"

The marketing says SSMs handle infinite context. The math says they handle long context via exponentially-decaying memory. Both are true — but the decay rate is fast enough that you can't blindly throw arbitrarily-long sequences at them and expect retention. Tune the state size and decay for your use case.

## Bottom line

For finance in 2026:

- **For < 1024-step sequences**: stick with TCN or small transformer.
- **For 1024-16384-step sequences**: SSMs (S4/Mamba) become attractive.
- **For training your own**: use the official `mamba-ssm` repo; the toy implementation above is for understanding only.

Continue to **[Neural ODEs and SDEs](05-neural-ode-sde.md)**.
