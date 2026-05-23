# Child-order placement with reinforcement learning

The Almgren-Chriss schedule tells you **how many shares** to trade at each step. It doesn't tell you **how to place each child order** — market vs limit, what limit price, at what venue. That choice is where most of the realised slippage variance comes from in modern execution.

This chapter is the practical RL recipe for the child-order placement problem. Builds on Module 13 (RL) and Module 19 chapter 5 (AC).

## The framing

For each AC-recommended slice (e.g., 5000 shares to trade in the next minute), the RL agent decides:

- **Order type**: market (pay the spread, fill immediately) or limit (save the spread, may not fill).
- **Limit price**: how far inside or outside the book.
- **Time-to-cancel**: how long to wait before pulling the order.

State: (shares remaining for the slice, time remaining for the slice, current bid/ask, recent flow signals, queue position estimate).

Reward: -(realised slippage vs the slice's arrival price). Larger is worse.

## A toy execution environment

```python
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ChildOrderEnv(gym.Env):
    """One AC slice: place limit orders, optionally cross to market."""
    metadata = {}

    def __init__(self, slice_size: int = 5000, time_steps: int = 60,
                 spread_bps: float = 2.0, sigma_per_step: float = 0.01,
                 fill_intensity_at_mid: float = 0.5, seed: int = 0):
        super().__init__()
        self.slice_size = slice_size; self.time_steps = time_steps
        self.spread_bps = spread_bps; self.sigma_per_step = sigma_per_step
        self.fill_intensity_at_mid = fill_intensity_at_mid
        self.rng = np.random.default_rng(seed)
        # Action: (limit_distance_bps, max_age_seconds, market_now_flag)
        # Discretise for simplicity: 5 levels × 3 ages × 2 = 30 actions
        self.action_space = spaces.Discrete(30)
        # Observation: (shares_remaining_pct, time_remaining_pct, mid_drift_bps, recent_fills_rate)
        self.observation_space = spaces.Box(-5, 5, shape=(4,), dtype=np.float32)
        self.mid = 100.0
        self._reset_state()

    def _reset_state(self):
        self.t = 0
        self.shares_remaining = self.slice_size
        self.cost = 0.0
        self.recent_fills = 0
        self.arrival_price = self.mid

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._obs(), {}

    def _obs(self):
        return np.array([
            self.shares_remaining / self.slice_size,
            (self.time_steps - self.t) / self.time_steps,
            (self.mid - self.arrival_price) / self.arrival_price * 10000,
            self.recent_fills,
        ], dtype=np.float32)

    def _decode_action(self, action):
        market = action % 2
        age = (action // 2) % 3                # 0, 1, 2 (steps to live)
        distance = (action // 6) % 5           # 0..4 (bps from mid)
        return market, age, distance

    def step(self, action):
        market, age, distance = self._decode_action(action)
        # Simulate mid drift
        self.mid += self.rng.normal(0, self.sigma_per_step)
        spread = self.mid * self.spread_bps / 10000
        ask = self.mid + spread / 2
        bid = self.mid - spread / 2
        if market or self.t == self.time_steps - 1:
            # Cross to market, fill at ask (buying)
            fill_price = ask
            fill_qty = self.shares_remaining
        else:
            # Limit at bid - distance * spread (further from market = less likely to fill)
            limit_price = bid - distance * spread / 10
            # Fill probability decays with distance from market
            p_fill = self.fill_intensity_at_mid * np.exp(-2.0 * distance)
            if self.rng.random() < p_fill:
                fill_qty = min(self.shares_remaining, 500)
                fill_price = limit_price
            else:
                fill_qty = 0; fill_price = 0.0
        self.cost += fill_qty * fill_price
        self.shares_remaining -= fill_qty
        self.recent_fills = fill_qty / self.slice_size
        self.t += 1
        terminated = self.shares_remaining <= 0 or self.t >= self.time_steps
        reward = 0.0
        if terminated:
            # Final reward: negative average slippage vs arrival
            avg_price = self.cost / self.slice_size
            reward = -(avg_price - self.arrival_price) / self.arrival_price * 10000
        return self._obs(), float(reward), bool(terminated), False, {}
```

A real env would use historical order-book data instead of synthetic dynamics; the simulator pattern is the same.

## Training PPO on the env

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def train_child_placement_agent(total_timesteps: int = 1_000_000):
    env = DummyVecEnv([lambda: ChildOrderEnv()])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO("MlpPolicy", env, n_steps=2048, batch_size=64,
                 learning_rate=3e-4, gamma=0.99, verbose=0)
    model.learn(total_timesteps=total_timesteps)
    return model, env
```

For the toy env, you can expect convergence in 100k-1M steps. The learned policy will:

- **Place mostly limit orders** at the bid when there's time remaining and recent flow has been good.
- **Cross to market** as time runs out, or when the mid drifts against you.
- **Use deeper limit prices** when recent fill rate has been high (lots of demand on your side).

## Production deployment

In production:

1. Train against a **historical replay simulator** (orderbook reconstruction from your actual broker's data).
2. **Validate** on out-of-sample dates.
3. **Paper trade** for 30+ days.
4. **Shadow deploy** alongside an AC-equal-slice baseline.
5. **Promote to live** only when shadow shows consistent slippage improvement.

Sample efficiency of paper trading is poor (only ~252 trading days per year per asset). Shadow deployment lets you accumulate samples while keeping risk low.

## Why this beats fixed strategies

A fixed "VWAP at the bid" strategy can't adapt. The RL agent learns:

- When to be patient (low VPIN, calm market, time available).
- When to be urgent (vol spike, recent fills slow, time short).
- When to use the dark pool vs the lit market (if your env exposes that).

Real production RL execution algos at large banks save 0.5-2 bps per trade vs the AC baseline.

## Pitfalls

!!! warning "Simulator fidelity"
    The agent learns from the simulator. A simulator that's wrong in a specific way creates an agent that exploits that wrongness. Sim-to-real gap (Module 13 chapter 6) is the dominant failure mode.

!!! warning "Reward hacking"
    If you penalise only realised cost, the agent may learn to cancel and re-place to game some metric. Add explicit penalties for cancel-rate and order-rate.

!!! warning "Adverse selection on limits"
    A limit fill is more likely to occur on toxic flow (the counterparty knew something). The agent should learn to widen quotes when toxic-flow indicators rise.

!!! warning "Multi-objective rewards**
    Cost vs information leakage vs fill ratio are different objectives. A scalar reward is a compromise; consider multi-objective RL (Pareto-optimal policies).

## Bottom line

RL on top of AC is the **modern execution stack**:

- **AC schedule** for the high-level (how many shares per minute).
- **RL agent for child-order placement** (limit vs market, where, how long).
- **Historical replay simulator** for training.
- **Shadow deployment** for production validation.

For most retail and small-fund work, this is overkill — TWAP / VWAP with a few hand-coded heuristics is plenty. For institutional execution at scale, RL is mainstream.

## End of Module 19

You now have the working execution stack: order books, microstructure scanners (Kyle's lambda, VPIN), noise-robust vol estimation (realised kernels), market-making (Avellaneda-Stoikov), and optimal execution (Almgren-Chriss + RL). The next module — **Risk and Portfolio Construction** — completes the institutional toolkit.

Continue to **[Module 20 — Risk and Portfolio Construction](../20-risk-and-portfolio/index.md)**.
