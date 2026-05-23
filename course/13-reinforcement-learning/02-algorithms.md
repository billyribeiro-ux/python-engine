# PPO, SAC, TD3 — the working algorithms

Three modern RL algorithms cover 90% of practical trading RL. PPO (Proximal Policy Optimization) is the safe default. SAC (Soft Actor-Critic) wins for continuous-action problems. TD3 (Twin Delayed DDPG) is the deterministic-continuous-control workhorse. This chapter is the decision guide.

## The taxonomy

| Algorithm | Action space | Sample efficiency | Stability | Common use |
|---|---|---|---|---|
| **PPO** | discrete + continuous | low | high | the safe default; sizing, discrete-execution |
| **SAC** | continuous | high | medium | market making, continuous execution |
| **TD3** | continuous | high | medium | continuous execution; less stable than SAC |
| DQN | discrete | medium | medium | low-dim discrete actions; rarely competitive vs PPO |
| A2C | both | low | high | smaller-scale baseline; PPO usually wins |

For most trading problems, the choice is **PPO if the action is discrete or low-dimensional continuous**, **SAC if continuous and you want good sample efficiency**.

## PPO in one paragraph

Proximal Policy Optimization (Schulman et al., 2017). Policy-gradient algorithm. The "proximal" part is a clipped surrogate objective that prevents the policy from changing too much in one update — which is the failure mode of vanilla policy gradient.

Concretely, at each iteration:

1. Roll out the current policy for $T$ steps; collect $(s_t, a_t, r_t, s_{t+1})$.
2. Compute advantages $\hat A_t$ using Generalized Advantage Estimation (GAE).
3. Update the policy by maximising the clipped objective:

$$
\mathcal{L}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) \hat A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat A_t) \right]
$$

where $r_t(\theta) = \pi_\theta(a_t | s_t) / \pi_{\theta_{old}}(a_t | s_t)$ is the importance ratio.

4. Update the value function with MSE.

The clip parameter $\epsilon$ (typically 0.2) is the only hyperparameter that really matters. Otherwise, defaults from `stable-baselines3` work.

## SAC in one paragraph

Soft Actor-Critic (Haarnoja et al., 2018). Off-policy actor-critic for continuous actions. The "soft" part adds an entropy term to the reward, encouraging exploration:

$$
J(\pi) = \mathbb{E}_\pi \left[ \sum_t \gamma^t (r_t + \alpha H(\pi(\cdot | s_t))) \right]
$$

The entropy coefficient $\alpha$ is auto-tuned in modern SAC. Off-policy → uses a replay buffer → much higher sample efficiency than PPO.

For continuous market-making or execution problems where each interaction is expensive (simulator runtime), SAC's sample efficiency is a real advantage.

## TD3 in one paragraph

Twin Delayed DDPG (Fujimoto et al., 2018). Like SAC but deterministic: the policy outputs a single action, not a distribution. Three tricks (twin critics, delayed policy updates, target policy smoothing) fix the issues in vanilla DDPG that made it unstable.

TD3 is slightly more sample-efficient than SAC on some benchmarks but trickier to tune for stability. In practice for trading, **SAC is preferred** unless you specifically need a deterministic policy.

## Using `stable-baselines3`

The standard library for RL in Python. Wraps all three algorithms (and many more) with a clean API.

```python
from stable_baselines3 import PPO, SAC
import gymnasium as gym

env = gym.make("YourTradingEnv-v0")

# PPO with sensible defaults
model = PPO("MlpPolicy", env, n_steps=2048, batch_size=64, n_epochs=10,
            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, verbose=0)
model.learn(total_timesteps=1_000_000)

# Predict
obs, _ = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
```

Key hyperparameters:

- **`n_steps`**: rollout length before each update. 2048 is the default; for episodic problems, set it to be a multiple of episode length.
- **`batch_size`**: minibatch size for PPO update. 64-256 is reasonable.
- **`gamma`**: discount. Tune to your horizon (Chapter 1).
- **`learning_rate`**: 3e-4 is the AdamW default; if training is unstable, lower it.

For SAC:

```python
model = SAC("MlpPolicy", env, buffer_size=1_000_000, batch_size=256,
            tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1,
            learning_rate=3e-4, ent_coef="auto", verbose=0)
```

The `buffer_size` is the replay buffer; for short episodes this can be the whole training set.

## Practical training tips

### 1. Normalise observations

```python
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

env = DummyVecEnv([lambda: gym.make("YourTradingEnv-v0")])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
```

`VecNormalize` running-normalises observations and rewards. RL is *much* more stable on normalised data; this is non-negotiable.

### 2. Vectorise environments

```python
from stable_baselines3.common.vec_env import SubprocVecEnv

env = SubprocVecEnv([lambda: gym.make("YourEnv-v0") for _ in range(8)])
```

Run 8 envs in parallel; collect 8× the experience per wall-clock second. For RL, throughput is everything.

### 3. Set a clear evaluation protocol

```python
from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
```

Always evaluate with `deterministic=True` so you measure the policy, not the exploration noise. For trading, evaluate across multiple time windows (walk-forward).

### 4. Don't trust training reward

The training reward includes exploration noise and entropy. Always also report eval reward on a held-out window.

### 5. Save checkpoints; resume training

```python
model.save("checkpoints/ppo_iter_1m")
model = PPO.load("checkpoints/ppo_iter_1m", env=env)
model.learn(total_timesteps=2_000_000, reset_num_timesteps=False)
```

RL runs take hours; crashes happen. Checkpoint every 100k steps.

## A worked example: PPO on a toy sizing env

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO

class ToySizeEnv(gym.Env):
    """Bandit-style: choose size in [-1, 1]; reward = size * next_return - cost."""
    metadata = {}
    def __init__(self, returns: np.ndarray):
        super().__init__()
        self.returns = returns
        self.observation_space = spaces.Box(-5, 5, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.i = 0
        self.prev_size = 0.0

    def _obs(self):
        idx = self.i
        return np.array([
            self.returns[idx-1] if idx > 0 else 0.0,
            np.std(self.returns[max(0, idx-20):idx]) if idx > 0 else 0.0,
            self.prev_size,
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.i = 1
        self.prev_size = 0.0
        return self._obs(), {}

    def step(self, action):
        size = float(np.clip(action[0], -1, 1))
        ret = self.returns[self.i] if self.i < len(self.returns) else 0.0
        cost = 5e-4 * abs(size - self.prev_size)
        reward = size * ret - cost
        self.prev_size = size
        self.i += 1
        terminated = self.i >= len(self.returns)
        return self._obs(), float(reward), terminated, False, {}


# Train
returns = np.random.default_rng(0).normal(0.0005, 0.01, size=5000).astype("float32")
env = ToySizeEnv(returns)
model = PPO("MlpPolicy", env, verbose=0, n_steps=512, batch_size=64,
            learning_rate=3e-4, gamma=0.99)
model.learn(total_timesteps=50_000)
```

This trains in a minute or two and learns a small positive position when recent returns are positive — exactly what you'd hand-code. The point is the framework, not the toy result.

## Pitfalls

!!! warning "Reward hacking"
    RL will find any loophole. If your reward has a per-step cost that grows linearly with steps, the agent learns to terminate as quickly as possible. Penalise time only when it's genuinely costly.

!!! warning "Sim-to-real gap"
    A policy that wins on a backtest simulator may lose live because the simulator missed something — adverse selection, queue position, slippage realism. Always paper-trade extensively before deploying RL policies.

!!! warning "Stochastic vs deterministic evaluation"
    PPO is a stochastic policy. Random seed differences across runs can produce wildly different final policies. Train 3-5 seeds; report median performance.

!!! warning "Reward normalisation hides progress"
    `VecNormalize(norm_reward=True)` makes the training reward look stationary, but it's normalised. To see real progress, periodically evaluate on unnormalised reward in the eval environment.

## Bottom line

For trading RL:

- **PPO** is the safe default. Use it first.
- **SAC** when actions are continuous and simulation is expensive.
- Always **normalise observations**, **vectorise envs**, **evaluate deterministically**.
- For toy problems: 100k-1M steps. For real: 10M-100M.

Continue to **[Custom gymnasium environments for trading](03-gym-envs.md)**.
