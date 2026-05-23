# Neural ODEs and SDEs

Most deep models work on discrete sequences. **Neural ODEs** (Chen et al., 2018) extend deep learning to **continuous time** — the hidden state evolves according to a differential equation parameterised by a neural network. **Neural SDEs** add stochastic noise, making them the right framework for modelling continuous-time random processes — exactly what financial prices are.

This is frontier-track material. For most strategies you don't need it. For options pricing, irregular-event modelling, and generative simulation of price paths, it's genuinely powerful.

## The Neural ODE

A standard ResNet block computes $h_{t+1} = h_t + f(h_t, \theta)$. As you stack more blocks, this looks like an Euler step of the ODE:

$$
\frac{dh(t)}{dt} = f(h(t), t, \theta)
$$

A **Neural ODE** takes this to the limit: the network *is* the right-hand side, and you integrate it with an off-the-shelf ODE solver. Training uses the adjoint method to backpropagate through the solver in $O(1)$ memory.

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, 64), nn.Tanh(),
            nn.Linear(64, hidden),
        )
    def forward(self, t, h):
        return self.net(h)


class NeuralODE(nn.Module):
    def __init__(self, n_features=1, hidden=32):
        super().__init__()
        self.embed = nn.Linear(n_features, hidden)
        self.func = ODEFunc(hidden)
        self.head = nn.Linear(hidden, n_features)
    def forward(self, x0, t):
        # x0: (batch, n_features) initial state
        # t: (n_times,) integration times
        h0 = self.embed(x0)
        h = odeint(self.func, h0, t)              # (n_times, batch, hidden)
        return self.head(h)                       # (n_times, batch, n_features)
```

The killer property: **irregular sampling is native**. Pass any sorted time vector to `odeint`, get the state at exactly those times. No fixed timestep, no interpolation hacks.

For trading: fundamentals release dates, news events, FOMC meetings — these arrive irregularly. A neural ODE models the continuous evolution between events without artificially binning.

## Neural CDEs — continuously updated

Kidger et al. (2020). **Neural Controlled Differential Equations** extend Neural ODEs to handle continuous *streams* of irregular observations:

$$
dh(t) = f(h(t), \theta) \, dX(t)
$$

where $X(t)$ is the (interpolated) observation path. The hidden state updates continuously as new data arrives. NCDEs are the deep learning analogue of the Kalman filter for non-linear, non-Gaussian systems.

The library: [`torchcde`](https://github.com/patrick-kidger/torchcde).

## Neural SDEs

A Neural ODE is deterministic. A Neural SDE adds a stochastic noise term:

$$
dh(t) = f(h(t), \theta) dt + g(h(t), \theta) dW(t)
$$

The drift $f$ and the diffusion $g$ are both neural networks; $dW$ is Brownian motion. Sample paths from the model by Euler-Maruyama integration.

For financial returns, this is exactly what Heston, SABR, and the stochastic-vol literature attempt to model parametrically — and what Neural SDEs do non-parametrically. You can fit a Neural SDE to historical returns and use it to **simulate synthetic price paths** with realistic distributional properties.

```python
import torch
import torch.nn as nn

class NeuralSDE(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.drift = nn.Sequential(nn.Linear(hidden, 64), nn.Tanh(), nn.Linear(64, hidden))
        self.diffusion = nn.Sequential(nn.Linear(hidden, 64), nn.Tanh(),
                                       nn.Linear(64, hidden), nn.Softplus())

    def sample_path(self, h0, dt, n_steps):
        h = h0
        path = [h]
        for _ in range(n_steps):
            mu = self.drift(h)
            sigma = self.diffusion(h)
            noise = torch.randn_like(h) * (dt ** 0.5)
            h = h + mu * dt + sigma * noise
            path.append(h)
        return torch.stack(path)
```

Training: typically by minimising the negative log-likelihood of observed paths under the SDE (signature-method-based losses are also popular — see Kidger et al.). The library: [`torchsde`](https://github.com/google-research/torchsde).

## When to use them

- **Neural ODE** when your observations are irregularly spaced but the underlying state evolves smoothly.
- **Neural CDE** for online updating with continuous streams.
- **Neural SDE** when you want to *generate* synthetic price paths with learned distributional properties.

For "predict tomorrow's SPY return," a Neural ODE is hilariously overkill. For "estimate the implied dynamics of an emerging-market currency given irregular intervention events," it might be the right tool.

## Deep hedging — the killer SDE application

Bühler, Gonon, Teichmann, Wood (2019). Train a neural network to **hedge an option** under transaction costs, where the underlying follows a learned (or assumed) SDE. The agent's policy maps state (price, time, position) to hedge action; the loss is the variance of the terminal P&L.

```python
# Pseudocode
for episode in range(N_EPISODES):
    path = sde.sample_path(initial_price, dt, n_steps)
    position = 0
    cost = 0
    for t in range(n_steps):
        action = hedger(state=(path[t], t, position))
        cost += transaction_cost(action - position, path[t])
        position = action
    terminal_payoff = max(path[-1] - strike, 0)
    pnl = cost + position * path[-1] - terminal_payoff
    loss = pnl.var()                              # minimise hedging error variance
```

Module 15 implements this end-to-end. Deep hedging beats classical delta-hedging under realistic costs in many regimes.

## Pitfalls

!!! warning "ODE solvers are slow"
    Each forward pass invokes a numerical solver — orders of magnitude slower than a feed-forward net. For inference latency requirements under ~10ms, stick to discrete models.

!!! warning "Adjoint method has its own pitfalls"
    For stiff ODEs or long integration intervals, the adjoint can be numerically unstable. `torchdiffeq` supports both adjoint and direct backpropagation; benchmark both for your problem.

!!! warning "Neural SDEs need MANY paths to train"
    Variance reduction matters. Use antithetic sampling, multilevel Monte Carlo, or fit via signature methods rather than direct MLE on a few paths.

!!! warning "Don't underestimate the parametric baseline"
    For options, calibrated Heston or SABR (Module 15) often beats a neural SDE in production. Use the neural version when classical models fundamentally fail.

## Bottom line

Neural ODEs and SDEs are powerful but niche. They're the right tool when:

- The data is **irregularly sampled**.
- You need **continuous-time modeling** without arbitrary binning.
- You want **generative** capabilities (synthetic paths).
- The problem **truly demands non-parametric drift/diffusion**.

For most quant work, simpler discrete models are faster, more interpretable, and good enough. Reach for the continuous-time tools when the discrete approximation visibly hurts.

Continue to **[Diffusion models for synthetic market paths](06-diffusion.md)**.
