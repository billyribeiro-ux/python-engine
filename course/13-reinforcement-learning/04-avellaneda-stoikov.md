# Avellaneda-Stoikov as RL

The Avellaneda-Stoikov (AS) market-making model (2008) is the canonical analytic solution to the market-maker's optimal-quoting problem. It assumes a Brownian-motion mid-price and Poisson-arrival orders against your quotes; under those assumptions, it gives a closed-form expression for the optimal bid-ask spread and the inventory-skew adjustment. Real markets violate the assumptions in interesting ways. RL is how you generalise.

## The AS model in one paragraph

Mid-price evolves as Brownian motion: $dS_t = \sigma dW_t$. You post a bid at $S_t - \delta^b$ and an ask at $S_t + \delta^a$. Buy and sell orders arrive as Poisson processes whose intensity decreases exponentially with quote distance:

$$
\lambda^{b}(\delta) = A e^{-\kappa \delta}, \quad \lambda^a(\delta) = A e^{-\kappa \delta}
$$

You hold inventory $q_t$. Maximise expected utility of terminal wealth $W_T - \gamma_\text{risk} q_T^2$ (mean-variance utility on inventory risk):

$$
V(t, q, S) = \max_{\delta^b, \delta^a} \mathbb{E}[\text{terminal wealth} - \gamma_\text{risk} q_T^2]
$$

The optimal quotes (for symmetric $A, \kappa$):

$$
\text{reservation price: } r_t = S_t - q_t \gamma_\text{risk} \sigma^2 (T - t)
$$
$$
\text{spread: } \delta^a + \delta^b = \gamma_\text{risk} \sigma^2 (T-t) + \frac{2}{\gamma_\text{risk}} \log\left(1 + \frac{\gamma_\text{risk}}{\kappa}\right)
$$

The reservation price is shifted away from the mid by an amount proportional to inventory — the more long you are, the lower your bid AND ask become (you want to skew toward selling). The spread is a sum of an inventory-risk term and a market-impact term.

In code:

```python
import numpy as np

def avellaneda_stoikov_quotes(S, q, t, T, sigma, gamma_risk, kappa, A):
    """Returns (bid_offset, ask_offset) from mid for the AS optimal quotes."""
    inventory_skew = q * gamma_risk * sigma**2 * (T - t)
    half_spread = 0.5 * gamma_risk * sigma**2 * (T - t) + (1 / gamma_risk) * np.log(1 + gamma_risk / kappa)
    bid_offset = inventory_skew + half_spread        # quote at S - bid_offset
    ask_offset = -inventory_skew + half_spread       # quote at S + ask_offset
    return bid_offset, ask_offset
```

Run this once at every quote update and you have an analytic market-making strategy.

## Where AS breaks

1. **The mid-price isn't pure Brownian motion.** Real prices have momentum, mean reversion, jumps.
2. **Arrival intensity isn't exponentially decreasing in distance.** Real flow has clustering, time-of-day patterns, and adverse selection that AS doesn't model.
3. **Adverse selection.** Toxic flow — informed traders who pick off your stale quotes — costs more than uninformed flow. AS treats all flow equally.
4. **Discreteness.** Tick sizes, lot sizes, minimum quote sizes. AS is continuous.
5. **Latency and queue position.** Your quote at the back of a long queue may not fill before the price moves.

All of these are why real market makers don't just run pure AS. They use AS as the *backbone* and learn corrections via RL.

## RL on top of AS

The pattern: use AS quotes as the **action prior** and let RL learn an adjustment based on additional state.

```python
def hybrid_policy(state, model_rl):
    # AS provides a quote suggestion
    bid_offset_as, ask_offset_as = avellaneda_stoikov_quotes(...)
    # RL learns a small additive correction
    bid_adj, ask_adj = model_rl.predict(state)
    return bid_offset_as + bid_adj, ask_offset_as + ask_adj
```

The RL model has to learn smaller corrections than learning the full policy from scratch — this dramatically improves sample efficiency and stability. It's a form of **residual RL**.

## A simplified market-making env

```python
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SimpleMMEnv(gym.Env):
    """Toy market-maker over a Brownian mid with Poisson arrivals."""
    def __init__(self, T=100, sigma=0.5, kappa=1.5, A=10.0, gamma_inv=0.001, max_q=10):
        super().__init__()
        self.T = T; self.sigma = sigma; self.kappa = kappa; self.A = A
        self.gamma_inv = gamma_inv; self.max_q = max_q
        # State: (mid, inventory, time_left)
        self.observation_space = spaces.Box(-100, 100, shape=(3,), dtype=np.float32)
        # Action: (bid_offset, ask_offset) — must be positive
        self.action_space = spaces.Box(0.001, 5.0, shape=(2,), dtype=np.float32)
        self.rng = np.random.default_rng(0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.mid = 100.0
        self.q = 0
        self.t = 0
        self.cash = 0.0
        return self._obs(), {}

    def _obs(self):
        return np.array([self.mid - 100.0, float(self.q), self.T - self.t], dtype=np.float32)

    def step(self, action):
        bid_off, ask_off = float(action[0]), float(action[1])
        # Simulate one time step
        # Brownian mid move
        self.mid += self.rng.normal(0, self.sigma)
        # Fill probabilities (per step, exponential intensity → discretised Bernoulli)
        p_buy = 1 - np.exp(-self.A * np.exp(-self.kappa * bid_off))   # someone hits our bid
        p_sell = 1 - np.exp(-self.A * np.exp(-self.kappa * ask_off))  # someone lifts our ask
        rng = self.rng
        bid_hit = rng.random() < p_buy and self.q < self.max_q
        ask_hit = rng.random() < p_sell and self.q > -self.max_q
        if bid_hit:
            self.q += 1; self.cash -= (self.mid - bid_off)
        if ask_hit:
            self.q -= 1; self.cash += (self.mid + ask_off)
        # Reward: change in mark-to-market wealth minus inventory penalty
        wealth = self.cash + self.q * self.mid
        reward = float(wealth) - self.gamma_inv * (self.q ** 2)
        # Reset reward to per-step delta instead of absolute
        if not hasattr(self, "_last_wealth"):
            self._last_wealth = 0.0
        delta_wealth = wealth - self._last_wealth
        self._last_wealth = wealth
        reward = delta_wealth - self.gamma_inv * (self.q ** 2)
        self.t += 1
        terminated = self.t >= self.T
        return self._obs(), float(reward), bool(terminated), False, {"wealth": float(wealth), "q": self.q}
```

This is a working market-making env for RL. Train PPO on it and compare the learned policy to the analytic AS quotes.

```python
from stable_baselines3 import PPO
env = SimpleMMEnv()
model = PPO("MlpPolicy", env, verbose=0).learn(total_timesteps=200_000)
```

## What RL adds beyond AS

In the simple env above with all assumptions satisfied, RL learns to approximate AS — and not much better. The gains come when you make the env **more realistic**:

- **Mean-reverting mid** instead of pure Brownian → RL learns to lean into the mean.
- **Adverse selection** (toxic flow hits the side that's about to be wrong) → RL learns to widen quotes when the predictive signal is strong.
- **Time-varying arrival intensity** → RL learns to quote tighter when liquidity demand is high.

In every one of these, AS underperforms because its closed form depends on assumptions that don't hold. RL learns the right adjustment from data.

## Pitfalls

!!! warning "RL on a clean-but-fake simulator"
    If your env is the AS world (Brownian mid, Poisson arrivals), RL just rediscovers AS. The whole point of RL here is the env departing from AS in realistic ways.

!!! warning "Inventory penalty too large"
    Crank `gamma_inv` and the agent learns to refuse to take inventory at all — quoting too wide to ever fill. Tune to match your real inventory tolerance.

!!! warning "Sample efficiency on long episodes"
    Real market-making sessions are millions of quote events. Don't simulate at that resolution unless you can run thousands of episodes in parallel. Often a 1000-step episode at coarser time-binning captures the dynamics.

!!! warning "Sim-to-real on adverse selection"
    The hardest thing to simulate is *informed* counterparties. RL trained on uninformed-flow sims systematically over-quote in production. Always paper-trade extensively.

## Bottom line

For market-making RL:

- **Start with AS as the analytic baseline.**
- Build a simulator that **departs from AS in ways that matter to your asset** (mean-reversion, adverse selection, time-of-day).
- **Train RL as residuals on top of AS quotes**, not from scratch.
- **Paper-trade extensively** before deploying — sim-to-real gap is the hardest in market-making.

Continue to **[Almgren-Chriss optimal execution](05-optimal-execution.md)**.
