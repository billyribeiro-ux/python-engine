# Meta-learning (MAML) for fast adaptation across symbols

A standard ML model trained on SPY-only data doesn't transfer well to AAPL — the dynamics differ. Training a separate model per ticker is wasteful (no sharing). Pooling all tickers into one model loses ticker-specific structure. **Meta-learning** is the framework that solves this: train one model that adapts *fast* to a new ticker, using only a few examples.

Model-Agnostic Meta-Learning (MAML, Finn et al. 2017) is the cleanest implementation. This chapter sketches MAML for finance.

## The idea

Standard training: minimise the loss across all training data, get the best model for the average task.

Meta-training: minimise the loss **after a few gradient steps of adaptation** on each new task. Get a model that is itself *easy to fine-tune* on new tasks.

The key insight: the meta-trained model isn't the best initial model — it's the model that **gets best fastest** after seeing a few new examples.

## The MAML algorithm

For tasks $\{T_i\}$ (each task = one ticker), training data $D_i$, and a model with parameters $\theta$:

1. For each task $T_i$, **inner update**: $\theta_i' = \theta - \alpha \nabla_\theta L_{T_i}(D_i^\text{train})$.
2. **Outer update**: $\theta \leftarrow \theta - \beta \nabla_\theta \sum_i L_{T_i}(D_i^\text{test} | \theta_i')$.

The outer update minimises loss *of the adapted model* — pushing $\theta$ toward a point where the inner update from any task leads to a good final model.

## A working implementation

```python
import torch
import torch.nn as nn
import copy


def maml_step(model, tasks_train, tasks_test, inner_lr=0.01, outer_optim=None):
    """One MAML outer step over a batch of tasks.
    Each task has (X_train, y_train, X_test, y_test) tensors."""
    outer_optim.zero_grad()
    losses_meta = []
    for X_tr, y_tr, X_te, y_te in zip(tasks_train, tasks_test, strict=False):
        # Inner adaptation
        fast_params = list(model.parameters())
        pred_tr = model(X_tr)
        loss_tr = torch.nn.functional.mse_loss(pred_tr, y_tr)
        grads = torch.autograd.grad(loss_tr, fast_params, create_graph=True)
        fast_params = [p - inner_lr * g for p, g in zip(fast_params, grads, strict=True)]
        # Forward with fast_params (need a functional pass; here we use copy as a sketch)
        adapted = copy.deepcopy(model)
        with torch.no_grad():
            for adapted_p, fast_p in zip(adapted.parameters(), fast_params, strict=True):
                adapted_p.copy_(fast_p)
        pred_te = adapted(X_te)
        loss_te = torch.nn.functional.mse_loss(pred_te, y_te)
        losses_meta.append(loss_te)
    meta_loss = sum(losses_meta) / len(losses_meta)
    meta_loss.backward()
    outer_optim.step()
    return meta_loss.item()
```

For real MAML, use a library like [`learn2learn`](https://github.com/learnables/learn2learn) which handles the functional gradient computation cleanly. The sketch above isn't directly trainable because `copy.deepcopy` breaks the gradient graph.

## A trading use case

Suppose you train a 3-layer MLP that takes 30 features (past returns, vol, etc.) and predicts the next 5-day return. Standard training pools data from all 500 stocks → one generalist model.

Meta-training: each "task" is one stock. The model is trained so that, after seeing 30 days of data from a new stock, a single SGD step adapts it to that stock's idiosyncrasies.

At deployment time: for a new IPO with 30 days of history, do one adaptation step and you have a working predictor — without retraining from scratch.

```python
# Schematic
class TradeMLP(nn.Module):
    def __init__(self, n_features=30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, X):
        return self.net(X).squeeze(-1)


# Tasks = stocks. For each, X_train is recent features, X_test is more-recent features.
# Meta-train across hundreds of stocks for many epochs.
# Deploy on a new stock by one inner-update gradient step.
```

## When MAML wins

- **Many related tasks** (hundreds of stocks).
- **Small per-task data** (a few months of history per stock).
- **Tasks share structure** but have idiosyncrasies.
- **Need fast adaptation** for new tasks.

For a global-equity panel where new stocks enter regularly, MAML is structurally the right tool. For "predict SPY's next return," it's overkill.

## Practical results

In the published literature, MAML on financial cross-section problems has shown 0.05-0.15 OOS Sharpe improvement over the pooled-baseline. The gains are bigger on **short-history tasks** (new IPOs, emerging-market stocks) where pooled training underweights the idiosyncrasies.

The cost: meta-training is heavy (second-order gradients), and the model is more complex to debug.

## Alternative: Reptile

Reptile (Nichol et al. 2018) approximates MAML with **only first-order gradients**:

1. Take K SGD steps on a single task to get $\theta_i^*$.
2. Update meta-parameters toward $\theta_i^*$ (instead of using gradients through the inner steps).

Mathematically cleaner; computationally cheaper; empirically nearly as good. The right choice for production.

```python
def reptile_step(model, task_data, inner_lr=0.01, n_inner_steps=5, outer_lr=0.1):
    """One Reptile outer step over a single task."""
    initial_params = [p.detach().clone() for p in model.parameters()]
    inner_optim = torch.optim.SGD(model.parameters(), lr=inner_lr)
    for _ in range(n_inner_steps):
        loss = nn.functional.mse_loss(model(task_data["X"]), task_data["y"])
        inner_optim.zero_grad(); loss.backward(); inner_optim.step()
    # Move meta-parameters toward task-adapted parameters
    for p, init in zip(model.parameters(), initial_params, strict=True):
        with torch.no_grad():
            p.data = init + outer_lr * (p.data - init)
    return loss.item()
```

For trading, Reptile is usually the right choice — simpler, faster, comparable results.

## Pitfalls

!!! warning "Task distribution at test time"
    MAML is trained for a particular task *distribution*. If you train on US large-caps and deploy on emerging-market microcaps, the meta-trained init is irrelevant.

!!! warning "Inner-step instability"
    The inner SGD step's learning rate is critical. Too small → no adaptation. Too large → overshooting. Tune it carefully (often through a meta-search).

!!! warning "Overfitting at meta level"
    The meta-model can memorise tasks. Use proper meta-CV (held-out tasks at meta-test time).

!!! warning "Time complexity**
    Each meta-step is O(N_tasks × N_inner_steps × forward+backward). For 500 stocks and 5 inner steps, that's 2500 forwards per outer step. Scale carefully.

## Bottom line

Meta-learning (MAML / Reptile) is the right tool when:

- You have **many related tasks** but limited data per task.
- You need **fast adaptation** to new tasks at deployment.
- The tasks share **transferable structure**.

For finance: a meta-trained cross-sectional model can absorb data from hundreds of stocks and adapt to new ones in seconds. Best paired with a downstream calibration / sizing layer (Modules 10-11).

## End of Module 17

You now have the frontier toolkit: Hawkes processes, TDA, transfer entropy, signatures, GNNs, diffusion-based scenarios, and meta-learning. None of these are silver bullets; all of them are research-grade ideas that occasionally produce alpha. The next module — **Scanners** — wraps the production and frontier ideas into an actionable framework.

Continue to **[Module 18 — Scanners](../18-scanners/index.md)**.
