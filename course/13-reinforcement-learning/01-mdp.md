# The MDP framing of trading

Reinforcement learning is the discipline of solving **Markov Decision Processes** (MDPs). Before you throw an RL library at a trading problem, it pays to first cast that problem as an MDP precisely — what is the state, what are the actions, what is the reward — because most "RL for trading" failures come from a sloppy MDP specification, not from the algorithm.

## The MDP, in five symbols

An MDP is a tuple $(S, A, P, R, \gamma)$:

- $S$ — the **state space**. Everything the agent observes at each step.
- $A$ — the **action space**. The set of actions available.
- $P(s' | s, a)$ — **transition dynamics**. The probability of landing in $s'$ given action $a$ in state $s$.
- $R(s, a, s')$ — **reward function**. Scalar feedback per transition.
- $\gamma \in [0, 1]$ — **discount factor**. How much you care about future vs immediate reward.

The agent's job is to learn a **policy** $\pi(a | s)$ that maximises the expected discounted cumulative reward.

## Casting a sizing problem as an MDP

Suppose you have a binary signal (long / flat / short) and want to learn how big to bet based on context (volatility, regime, recent P&L). The MDP:

- **State**: signal_today, vol_today, drawdown_so_far, days_since_last_loss, regime_indicator (etc.).
- **Action**: position size, discretised into N levels (e.g., {0%, 25%, 50%, 75%, 100%} of max).
- **Transition**: market moves; next state is computed.
- **Reward**: per-bar log-return × position − transaction cost.
- **Discount**: 0.999 for daily bars (weighs ~1 year out).

The policy: given today's state, choose today's size.

## Two harder framings

### Execution

- **State**: time remaining, shares remaining, current bid/ask, recent prints.
- **Action**: how many shares to send next, and at what limit / market.
- **Reward**: -cost (cost = avg fill price - arrival price, scaled by total shares).
- **Discount**: 1.0 (we care about total cost, not when in the schedule).
- **Horizon**: typically minutes to hours.

### Market making

- **State**: inventory, time left in session, mid-price, recent volatility, queue position estimate.
- **Action**: (bid_offset, ask_offset, bid_size, ask_size).
- **Reward**: realised spread captured − inventory penalty − adverse-selection cost.
- **Discount**: depends on session length.

Both are classical optimisation problems with analytical solutions in simplified cases (Avellaneda-Stoikov, Almgren-Chriss). RL is the right tool when the environment is more complex than the analytics handles.

## Why trading violates standard MDP assumptions

Three big ways markets are not nice MDPs:

### 1. Non-stationarity

Standard MDP assumes the transition function is fixed. Markets clearly evolve — volatility regimes shift, microstructure changes with technology, participants change behaviour. A policy trained on 2018 may underperform on 2024.

**The fix**: include regime indicators in the state, retrain periodically, monitor for distribution shift, use ensembles across training windows.

### 2. Partial observability

You can never observe the full state — other participants' positions, hidden orders, news arrivals are unobservable. This is a **POMDP** (Partially Observable MDP), not an MDP.

**The fix**: include history in the state (last N bars), or use a recurrent policy that builds its own state from observations.

### 3. Your own market impact

Standard MDP assumes your actions don't change the transition dynamics. For small accounts that's roughly true. For institutional sizes, it's false — your trade moves the market, your future transitions depend on your past trades.

**The fix**: in simulation, include a market-impact model (Almgren-Chriss style, Module 9 chapter 6). In live, throttle exposure to keep impact small.

## Reward shaping — the most important MDP choice

The reward function defines what the agent optimises. Pick the wrong reward and the agent finds clever ways to exploit it.

Common mistakes:

- **Reward = total P&L**, with no risk penalty → agent learns to take huge leverage.
- **Reward = Sharpe** → agent learns to deliver mediocre P&L with very low variance, missing opportunities.
- **Reward = next-bar return** with no discount → agent learns short-horizon scalping that doesn't compound.

A reasonable starting reward:

$$
r_t = \text{log-return}_t \times \text{position}_t - \lambda |\Delta \text{position}_t| - \mu \cdot (\text{drawdown}_t)
$$

The transaction-cost term ($\lambda$) discourages over-trading. The drawdown penalty ($\mu$) discourages risk-taking that endangers the account.

Tune $\lambda$ and $\mu$ to match your real-world constraints. RL is brutally efficient at exploiting whatever objective you give it; the objective must be correct.

## Discount factor matters

For daily bars, $\gamma = 0.999$ effectively averages reward over ~1000 days. For minute bars, $\gamma = 0.999$ is ~17 hours — too short. For tick data, $\gamma = 0.999999$ is more sensible.

The discount controls horizon. If your strategy is supposed to be 1-month, $\gamma = 0.95$ for daily bars (~20-day effective horizon). If it's annual, $\gamma = 0.998$.

## Episodes vs continuing tasks

- **Episodic**: each "trial" is one bounded period (a trading day, a single big order, one calendar month). Reset state between episodes.
- **Continuing**: one infinite stream (sizing across years).

Standard RL libraries are tuned for episodic tasks. For continuing problems, you usually still split into "episodes" of fixed length for stability, even if conceptually the task continues.

## The simulator question

RL is hungry for samples — typically tens of millions of agent-environment interactions. Where do those come from?

- **Bootstrap-style replay** on historical data. The agent acts on past bars; the "next state" comes from history. Cheap; doesn't capture market impact.
- **Simulator with a learned price model.** Train a generative model (Module 12) on history; use it as the environment. Can over-fit to the simulator's biases.
- **Live paper trading.** Slow, but the only way to get truly fresh data. Good for fine-tuning a policy trained offline.

The course's approach: **offline RL on bootstrap-replay data**, then paper-trade for final tuning. Pure online RL against the real market is generally too slow.

## A worked sketch: episode definition for a daily sizing problem

```python
import numpy as np
import pandas as pd

class SizingEpisode:
    """One trading "month" of bars; agent decides size each day."""
    def __init__(self, bars: pd.DataFrame, features: pd.DataFrame, start: int, length: int = 21):
        self.bars = bars.iloc[start:start + length]
        self.features = features.iloc[start:start + length]
        self.equity = 1.0
        self.position = 0.0
        self.i = 0
        self.drawdown = 0.0
        self.peak = 1.0
    def state(self):
        return np.concatenate([
            self.features.iloc[self.i].values,
            [self.position, self.drawdown, self.equity],
        ])
    def step(self, action: float):
        # action is the new target position in [-1, 1]
        prev_pos = self.position
        self.position = float(np.clip(action, -1, 1))
        ret = self.bars["close"].pct_change().iloc[self.i + 1] if self.i < len(self.bars) - 1 else 0.0
        cost = 1e-4 * abs(self.position - prev_pos)
        pnl = self.position * ret - cost
        self.equity *= (1 + pnl)
        self.peak = max(self.peak, self.equity)
        self.drawdown = self.equity / self.peak - 1
        reward = np.log(1 + pnl) - 5.0 * max(-self.drawdown - 0.05, 0)
        self.i += 1
        done = self.i >= len(self.bars) - 1
        return self.state(), reward, done
```

Twenty lines. A complete sizing-as-RL episode. The interesting choices are in the reward function (log return + drawdown penalty) and in the action shaping (continuous clipped to [-1, 1]).

## When NOT to use RL

For "predict the next-bar return and trade proportional to the prediction," RL is overkill. Use a supervised model (Module 10), calibrate it (Module 10 ch4), size by conformal interval (Module 10 ch5). Three concrete steps; no MDP needed.

RL pays when the **action affects future state** non-trivially — execution scheduling, market making, online sizing under drawdown stops. For pure prediction tasks, supervised wins on data efficiency.

## Bottom line

For RL on trading:

- **Cast the problem as an MDP rigorously** before reaching for an algorithm.
- **Reward = P&L − transaction cost − drawdown penalty** is a working starting point; tune coefficients.
- **State must include enough of the history** to recover the Markov property (use a recurrent policy if a fixed-size state isn't enough).
- **Discount factor matches the strategy horizon**.
- **Train offline on replay; fine-tune live on paper trading**.

Continue to **[PPO, SAC, TD3 — the working algorithms](02-algorithms.md)**.
