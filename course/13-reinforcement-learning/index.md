# Module 13 — Reinforcement Learning

Most of this course teaches *prediction*: given features, forecast the next return. RL teaches *decisions*: given state, choose an action that maximises cumulative reward. For execution (when and how to break up a big order), market making (where and how big to quote), and adaptive sizing (how to scale up after a winning streak), RL is the natural framework.

Pages:

1. **[The MDP framing of trading](01-mdp.md)** — states, actions, rewards, the discount; what's hard about applying it to markets.
2. **[PPO, SAC, TD3 — the working algorithms](02-algorithms.md)** — what each does, when each wins.
3. **[Custom gymnasium environments for trading](03-gym-envs.md)** — building an env for a sizing problem.
4. **[Avellaneda-Stoikov as RL](04-avellaneda-stoikov.md)** — the canonical market-making problem.
5. **[Almgren-Chriss optimal execution](05-optimal-execution.md)** — analytic schedule + RL extension.
6. **[Offline RL and the sim-to-real gap](06-offline-rl.md)** — when historical data is all you have.

Start with **[The MDP framing of trading](01-mdp.md)**.
