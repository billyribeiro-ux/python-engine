# Custom gymnasium environments for trading

A `gymnasium.Env` is the contract between your training algorithm and your simulation. Get the contract right and you can swap algorithms freely (PPO today, SAC tomorrow). Get it wrong and you waste compute training on a flawed environment.

This chapter is the working pattern for building a trading env, with a complete runnable example.

## The Env contract

A `gymnasium.Env` subclass must implement:

- **`__init__`** — set `observation_space` and `action_space`.
- **`reset(*, seed, options)`** — start a new episode; return `(obs, info)`.
- **`step(action)`** — apply action; return `(obs, reward, terminated, truncated, info)`.

That's it. Optional:

- **`render()`** — for visualisation.
- **`close()`** — clean up.

## A daily-bar sizing environment

```python
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class DailySizeEnv(gym.Env):
    """Episode = one fixed-length window of daily bars.
    Action = target position in [-1, +1].
    Reward = position-shifted return - turnover cost - drawdown penalty.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        bars: pd.DataFrame,
        features: pd.DataFrame,
        episode_len: int = 252,
        cost_per_turn: float = 1e-4,
        dd_penalty: float = 5.0,
        max_dd: float = 0.20,
        seed: int | None = None,
    ):
        super().__init__()
        assert bars.index.equals(features.index)
        self.bars = bars
        self.features = features
        self.episode_len = episode_len
        self.cost = cost_per_turn
        self.dd_pen = dd_penalty
        self.max_dd = max_dd
        self.rng = np.random.default_rng(seed)
        # Observation: features + (position, drawdown)
        self.obs_dim = features.shape[1] + 2
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._start = 0
        self._i = 0
        self._position = 0.0
        self._equity = 1.0
        self._peak = 1.0

    def _obs(self) -> np.ndarray:
        f = self.features.iloc[self._start + self._i].to_numpy(dtype=np.float32)
        dd = self._equity / self._peak - 1.0
        return np.concatenate([f, np.array([self._position, dd], dtype=np.float32)])

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        max_start = len(self.bars) - self.episode_len - 1
        self._start = int(self.rng.integers(0, max_start))
        self._i = 0
        self._position = 0.0
        self._equity = 1.0
        self._peak = 1.0
        return self._obs(), {}

    def step(self, action: np.ndarray):
        target = float(np.clip(action[0], -1.0, 1.0))
        prev_position = self._position
        self._position = target
        # The bar we act on is the NEXT bar's return
        idx = self._start + self._i + 1
        if idx >= len(self.bars):
            return self._obs(), 0.0, True, False, {}
        ret = float(self.bars.iloc[idx]["close"] / self.bars.iloc[idx - 1]["close"] - 1)
        turnover = abs(self._position - prev_position)
        pnl = self._position * ret - self.cost * turnover
        self._equity *= 1 + pnl
        self._peak = max(self._peak, self._equity)
        dd = self._equity / self._peak - 1.0
        reward = np.log1p(pnl) - self.dd_pen * max(-dd - self.max_dd, 0.0)
        self._i += 1
        terminated = (self._i >= self.episode_len) or (dd < -self.max_dd)
        return self._obs(), float(reward), bool(terminated), False, {
            "equity": self._equity, "drawdown": dd, "position": self._position
        }
```

Walk-through:

- **Observation space** is the feature vector concatenated with `(position, drawdown)`. The action depends on the current position (turnover) and on the drawdown (the drawdown penalty); the agent needs both in the state.
- **Action space** is a single continuous value in `[-1, 1]`.
- **Reset** picks a random episode start. This is important — fixed-start episodes lead to the agent memorising bar-by-bar; random-start forces generalisation.
- **Step** computes the realised return on the *next* bar, charges turnover cost, computes drawdown, returns a reward that combines per-bar log-return with a drawdown penalty.
- **Termination** on either episode length or breach of max drawdown.

## Wiring it to PPO

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# bars and features should be your prepared DataFrames
def make_env():
    return DailySizeEnv(bars=bars_train, features=features_train, episode_len=252, seed=0)

vec_env = DummyVecEnv([make_env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

model = PPO("MlpPolicy", vec_env, n_steps=2048, batch_size=64,
            learning_rate=3e-4, gamma=0.999, gae_lambda=0.95, clip_range=0.2,
            verbose=0)
model.learn(total_timesteps=500_000)
```

For evaluation, build a separate env with held-out data and disable normalisation updates:

```python
eval_env = VecNormalize(DummyVecEnv([lambda: DailySizeEnv(bars=bars_test, features=features_test)]),
                        training=False, norm_obs=True, norm_reward=False,
                        obs_rms=vec_env.obs_rms)
```

The `obs_rms=vec_env.obs_rms` reuses the training-time observation statistics — essential for consistency.

## Best-practice env design

### 1. Episode shorter than the strategy horizon

For a strategy that targets year-scale Sharpe, episodes of 252 days (1 trading year) are about right. Longer → fewer episodes per epoch → slower training. Shorter → agent can't learn long-horizon trade-offs.

### 2. Random episode starts

Always randomise. Otherwise the agent overfits to the specific path.

### 3. Use absolute features in the observation

Include both *level* features (current vol, current trend) and *relative* features (today's return vs 60-day mean). The agent needs both.

### 4. Don't include future information

Sounds obvious; happens constantly. Audit every feature: is it computed using only data with timestamp $\leq$ the bar being decided on? If you're unsure, write the env explicitly with `shift(1)`.

### 5. Penalise the actions you don't want

Want low turnover? Add a per-turn cost. Want to avoid huge drawdowns? Add a drawdown penalty. RL exploits whatever you reward; if you only reward P&L, you get a P&L-maximiser that may also be a leverage-maximiser.

### 6. Vectorise

```python
from stable_baselines3.common.vec_env import SubprocVecEnv

env = SubprocVecEnv([make_env for _ in range(8)])
```

8 parallel envs → 8× throughput. RL is bottlenecked by env sampling for most realistic simulations.

## Common bugs

!!! warning "Out-of-bounds indexing in step()"
    If `self._i >= len(self.bars) - self._start`, you'll get an `IndexError`. Always check before accessing the next bar.

!!! warning "Reward scale wildly different across episodes"
    If one episode's reward sum is 0.1 and another's is 100, training is unstable. Per-bar log-return clipped to [-1, 1] is usually sane; for very high vol regimes, scale further.

!!! warning "Observation that includes the action's effect"
    If your observation includes "today's P&L" computed from "today's action," the agent can see its action in the next state and gets confused. Either delay by one step or split state into "before action" and "after action" parts cleanly.

!!! warning "Episodes that always end at a profit"
    If your episode termination is "drawdown < X", but X is so large it never triggers, your reward shape is misleading. Test what fraction of episodes terminate by drawdown vs length.

## Testing the env

Before training, sanity-check the env:

```python
env = DailySizeEnv(bars, features)
obs, _ = env.reset(seed=0)
print(f"obs shape: {obs.shape}, dtype: {obs.dtype}")

# Run with random actions
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
print(f"random policy ended with equity {info['equity']:.3f}")
```

A random policy should produce equity near 1.0 (sometimes up, sometimes down). If it always crashes the env, you have a bug.

## Bottom line

For a working trading env:

- Subclass `gymnasium.Env`; implement reset and step honestly.
- Include position and drawdown in the observation.
- Randomise episode starts.
- Penalise turnover and excessive drawdown explicitly.
- Vectorise for throughput, normalise for stability.

The env design is half the RL game. Take your time on it; the algorithm is almost a commodity by comparison.

Continue to **[Avellaneda-Stoikov as RL](04-avellaneda-stoikov.md)**.
