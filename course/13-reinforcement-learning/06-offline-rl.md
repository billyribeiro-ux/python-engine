# Offline RL and the sim-to-real gap

Online RL needs an environment the agent can poke at — typically a simulator. For trading, the simulator is *always* an approximation: it doesn't model your impact, doesn't model adverse selection well, doesn't capture regime shifts. The gap between simulator and reality (sim-to-real) is the dominant source of RL deployment failures.

**Offline RL** is the alternative: learn from a fixed dataset of past interactions. No new sampling, no simulator. For trading, this often means learning from your own broker's order log — what actions did you take, what outcomes did you get, and what would have been better.

## The challenge of offline RL

Offline RL is harder than supervised learning *and* harder than online RL. Why:

1. **Distribution shift.** The behavioural policy that generated your data didn't necessarily explore all states. The learned policy may want to take actions far from what's in the dataset, and you have no data to evaluate those.
2. **Bootstrapping error.** Q-learning bootstraps estimates of $Q(s', a')$ from the next state. If $(s', a')$ is rarely in the data, the estimate is noisy and the error compounds.
3. **No environment for evaluation.** You can't just "run the policy" to see how it does. Off-policy evaluation (OPE) is its own hard problem.

The modern algorithms that work in this regime: **CQL** (Conservative Q-Learning), **IQL** (Implicit Q-Learning), **TD3+BC** (TD3 with behaviour cloning regularisation).

## CQL — penalise out-of-distribution actions

Kumar et al. (2020). Modifies the Q-learning loss to **underestimate** the value of actions not in the dataset:

$$
\mathcal{L}_\text{CQL} = \alpha \left( \mathbb{E}_{s \sim D, a \sim \pi}[Q(s, a)] - \mathbb{E}_{s, a \sim D}[Q(s, a)] \right) + \mathcal{L}_\text{TD}
$$

The first term penalises high Q-values for actions sampled from the current policy (which may be out-of-distribution); the second rewards high Q-values for in-distribution actions. The result: a Q-function that doesn't overestimate the value of unfamiliar actions.

## IQL — never query out-of-distribution

Kostrikov et al. (2021). Avoids OOD queries entirely by using only state-value function $V(s)$ updates that don't require evaluating $Q(s, a')$ for $a'$ from the policy. Much simpler to implement than CQL and frequently as effective.

```python
# Schematic — see official implementations for the full algorithm
import torch

class IQL:
    def __init__(self, q_network, v_network, policy_network, tau=0.7, beta=3.0):
        self.q = q_network; self.v = v_network; self.pi = policy_network
        self.tau = tau               # expectile for value loss
        self.beta = beta             # temperature for policy extraction

    def expectile_loss(self, target, pred, tau):
        diff = target - pred
        weight = torch.where(diff > 0, tau, 1 - tau)
        return (weight * diff ** 2).mean()

    def update(self, s, a, r, s_next):
        # Value update: V learns the tau-expectile of Q(s, a)
        with torch.no_grad():
            q_target = self.q(s, a)
        v_pred = self.v(s)
        v_loss = self.expectile_loss(q_target, v_pred, self.tau)

        # Q update: standard Bellman with V (not Q) as next-state value
        with torch.no_grad():
            v_next = self.v(s_next)
        q_target = r + 0.99 * v_next
        q_loss = ((self.q(s, a) - q_target) ** 2).mean()

        # Policy extraction: weighted log-likelihood with advantage weights
        with torch.no_grad():
            adv = self.q(s, a) - self.v(s)
            weight = torch.clamp(torch.exp(self.beta * adv), max=100)
        policy_loss = -(weight * self.pi.log_prob(a, s)).mean()
        return v_loss, q_loss, policy_loss
```

IQL doesn't query the policy for actions during the update — it works purely from the data. That's what makes it robust to distribution shift.

## A trading-specific offline RL recipe

For learning a sizing or execution policy from historical strategy logs:

1. **Collect** `(state, action, reward, next_state)` tuples from past trading decisions. Each row is one bar where you decided something.
2. **Choose IQL** (simpler) or **CQL** (more aggressive constraint).
3. **Tune the conservatism** — too strict and the policy is just behavioural cloning; too loose and you fall back to OOD-exploiting Q-learning.
4. **Evaluate on a held-out time window**, never on a different action distribution.
5. **Deploy cautiously** — paper-trade for at least 30 days before going live.

## Conservative Q-learning warning

A common deployment failure: CQL trained on logs where you only took 3 distinct action values produces a policy that picks one of those 3 values. The "policy" is essentially a re-weighting of behavioural cloning. Sometimes that's fine; sometimes it means CQL added no value over the simpler approach.

Test: does the trained CQL policy genuinely deviate from behavioural cloning? Run both on the eval data and compare action distributions.

## Off-policy evaluation

The Achilles heel of offline RL. To estimate "how good is this learned policy" without running it live, you need OPE methods:

- **Importance sampling (IS)** — re-weight observed rewards by the policy-to-behavioural ratio. Unbiased but very high variance.
- **Per-step IS, weighted IS, doubly-robust** — variance-reduction variants.
- **FQE (Fitted Q Evaluation)** — train a Q-function on observed transitions; estimate value of a candidate policy by averaging over its actions. Lower variance but model-dependent.

For trading, OPE estimates are noisy enough that the *only* honest pre-deployment check is paper trading.

## Sim-to-real techniques

When you must use a simulator (because you don't have enough log data), narrow the gap:

1. **Domain randomisation** — train the policy across many simulator variants (different volatility, impact, latency). The policy that wins across all is more robust to the real one.
2. **System identification** — fit the simulator parameters to your actual broker's behaviour (executed price vs quote, fill latency distributions).
3. **Adversarial training** — train against an adversary that perturbs the environment to find policy weaknesses.
4. **Curriculum** — start simple (no adverse selection), gradually add realism.

## Pitfalls

!!! warning "Logged data doesn't cover all states"
    If your historical strategy never took action X in state S, the offline RL agent can't credibly evaluate (S, X). It will either avoid that pairing (CQL) or hallucinate a value (vanilla Q-learning). Be honest about coverage.

!!! warning "Stale logs"
    Markets change. A policy trained on 2020-2022 logs deployed in 2025 will perform like the old policy adapted to old conditions — not necessarily what you want in 2025 markets.

!!! warning "Reward labelling errors"
    Offline RL's reward labels come from your past records. If those have bookkeeping errors (wrong sign on fees, missing dividends, mislabeled rebates), the agent learns to exploit them.

!!! warning "Discount factor mismatch"
    The discount factor at training time must match the deployment horizon. A policy trained for short-horizon (γ=0.9) deployed at long-horizon (γ=0.999) is operating in a different optimisation regime.

## Bottom line

For RL on trading:

- **Online RL** when you have a trustworthy simulator (rare; build one carefully).
- **Offline RL** when you have a large log of past decisions (more common; use IQL).
- **Hybrid** — train offline, deploy in paper trading, collect new on-policy data, retrain. Iterate.
- **OPE is noisy** — paper-trading is the only honest pre-deployment check.

## End of Module 13

You now have the RL toolkit for the trading problems where actions matter: sizing, execution, market-making. The next module — **Options Foundations** — is where we leave pure equities behind and start dealing with derivatives. Black-Scholes, the full Greeks, American exercise, and the implied-volatility machinery that everything else in options-land is built on top of.

Continue to **[Module 14 — Options Foundations](../14-options-foundations/index.md)**.
