# Graph neural networks on cross-asset correlation graphs

A graph neural network (GNN) is a deep model that operates on graph-structured data. For a universe of assets where edges encode pairwise relationships (correlation, sector membership, supply-chain links), a GNN can learn representations that simultaneously incorporate each asset's features AND its neighbours' features. For cross-sectional return prediction in a panel of thousands of stocks, that's structurally what you want.

This chapter sketches the architecture and shows the working setup.

## When a GNN beats a vanilla cross-sectional model

A vanilla XGBoost on cross-sectional features treats each stock independently — its features go in, a prediction comes out. The model doesn't know which stocks are related.

A GNN explicitly does: each node's prediction is a function of its own features AND the features of its neighbours, multiple layers deep. For tasks where peer-stock dynamics matter (mean reversion in pairs, sector-level information diffusion, supply-chain effects), GNNs can add real edge.

## The basic GNN layer

A graph convolutional network (GCN) layer, simplified:

$$
h_v^{(l+1)} = \sigma\left( W^{(l)} \cdot \frac{1}{|\mathcal{N}(v) \cup \{v\}|} \sum_{u \in \mathcal{N}(v) \cup \{v\}} h_u^{(l)} \right)
$$

Each node's new representation is a (learned) weighted average of its neighbours' (including itself's) old representations, then a non-linearity. Stack L layers and each node sees a neighbourhood of radius L.

```python
import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)
    def forward(self, X, A_norm):
        """X: (n_nodes, in_dim); A_norm: (n_nodes, n_nodes) row-normalised adjacency."""
        return torch.relu(self.W(A_norm @ X))


class SimpleGCN(nn.Module):
    def __init__(self, n_features, hidden=64, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [n_features] + [hidden] * (n_layers - 1) + [1]
        for i in range(n_layers):
            self.layers.append(GCNLayer(dims[i], dims[i+1]))

    def forward(self, X, A_norm):
        for layer in self.layers[:-1]:
            X = layer(X, A_norm)
        return self.layers[-1].W(A_norm @ X).squeeze(-1)     # final output without ReLU
```

For production, use [`torch_geometric`](https://pytorch-geometric.readthedocs.io/) which has efficient implementations of many GNN flavours (GCN, GAT, GraphSAGE, GIN).

## Building the graph

For an asset universe, three reasonable choices of edges:

1. **Correlation-thresholded** — connect pairs with rolling correlation > 0.5. Symmetric, undirected.
2. **k-nearest-neighbours by correlation** — connect each node to its top-k correlated peers. Sparse, directed.
3. **Hierarchical clustering** — connect within-cluster nodes. Block-sparse.

For a 500-asset universe, k-NN with k=20 gives ~10,000 edges — manageable for a GNN. Refresh the graph monthly.

```python
import numpy as np

def correlation_knn_adjacency(returns: pd.DataFrame, k: int = 20) -> np.ndarray:
    corr = returns.corr().values
    np.fill_diagonal(corr, -1)         # exclude self
    knn_idx = np.argsort(corr, axis=1)[:, -k:]    # top-k correlated
    n = corr.shape[0]
    A = np.zeros((n, n))
    for i in range(n):
        A[i, knn_idx[i]] = 1.0
    return A


def normalise_adjacency(A: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
    if add_self_loops:
        A = A + np.eye(len(A))
    D = A.sum(axis=1, keepdims=True)
    return A / np.where(D > 0, D, 1.0)
```

## A worked training loop

```python
import torch
import torch.nn as nn

def train_gnn(node_features_history, target_returns, A_norm,
               hidden=64, n_layers=3, epochs=100, lr=1e-3):
    """node_features_history: list of (n_nodes, n_features) over training dates.
       target_returns: list of (n_nodes,) over training dates.
       Same A_norm for all (assumes static graph). For time-varying, refit graph periodically."""
    model = SimpleGCN(n_features=node_features_history[0].shape[1],
                       hidden=hidden, n_layers=n_layers)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    A_t = torch.from_numpy(A_norm).float()
    for ep in range(epochs):
        model.train()
        ep_loss = 0
        for X_arr, y_arr in zip(node_features_history, target_returns):
            X = torch.from_numpy(X_arr).float()
            y = torch.from_numpy(y_arr).float()
            pred = model(X, A_t)
            loss = loss_fn(pred, y)
            optim.zero_grad(); loss.backward(); optim.step()
            ep_loss += loss.item()
        if ep % 10 == 0:
            print(f"Epoch {ep}: loss = {ep_loss / len(node_features_history):.4f}")
    return model
```

For each historical date, you have node features (e.g., 30 per-stock features) and target returns. The GNN predicts per-stock target from the joint structure.

## A more powerful variant: GAT (Graph Attention Networks)

GCN treats all neighbours equally. **Graph attention networks** (GAT) learn per-edge weights via attention:

$$
\alpha_{vu} = \text{softmax}_u(\text{attention}(h_v, h_u))
$$

The neighbour weighting becomes learned, not fixed. For asset graphs, GAT often outperforms GCN because some neighbours matter more than others depending on the regime.

`torch_geometric.nn.GATConv` implements this directly.

## GraphSAGE — for inductive learning

The GCN above is *transductive*: it works on a fixed graph. For real markets, new stocks join the universe, old ones leave. **GraphSAGE** trains aggregation functions that generalise to unseen nodes — making the model inductive.

For evolving universes, GraphSAGE is the right choice.

## A worked example: cross-sectional return prediction

```python
import pandas as pd
from engine.backtest import WalkForward

# Pretend we have:
# returns_panel: T x N daily returns
# features_panel: T x N x F daily per-asset features

returns_panel = ...
features_panel = ...

# Build a single static graph from the most recent year of correlations
A = correlation_knn_adjacency(returns_panel.tail(252), k=20)
A_norm = normalise_adjacency(A)

# Walk-forward train
wf = WalkForward(initial_train=252 * 2, test_size=21, embargo=5)
preds = pd.DataFrame(index=returns_panel.index, columns=returns_panel.columns, dtype=float)
for tr_idx, te_idx in wf.split(len(returns_panel)):
    train_X = [features_panel[t].values for t in tr_idx[:-5]]
    train_y = [returns_panel.shift(-5).iloc[t].values for t in tr_idx[:-5]]
    model = train_gnn(train_X, train_y, A_norm, epochs=50)
    model.eval()
    for t in te_idx:
        X = torch.from_numpy(features_panel[t].values).float()
        with torch.no_grad():
            preds.iloc[t] = model(X, torch.from_numpy(A_norm).float()).numpy()
```

Use the OOS predictions as a cross-sectional signal: long top decile, short bottom decile, dollar-neutral.

## Pitfalls

!!! warning "Graph stability"
    Correlations change. A graph built on 2022 data may not reflect 2024's relationships. Rebuild the graph at training time and consider it a hyperparameter.

!!! warning "Over-smoothing"
    Too many GNN layers (5+) make all nodes look alike — every prediction collapses to the same value. Stick to 2-4 layers for most asset-graph tasks.

!!! warning "Compute scaling"
    GCN on a 5000-asset graph is $O(N^2)$ in memory. For very large universes, use sparse adjacency representations and mini-batch training over subgraphs.

!!! warning "Edge weights from correlation aren't time-stationary"
    A high-correlation pair may have been low-correlation a year ago. Either accept this as the model's "current view" or use dynamic graphs (one per time step) — more compute, more flexibility.

## Bottom line

GNNs are the right tool when:

- You have a **large panel** of related assets (>200).
- The **relationships are explicit and meaningful** (correlation, sector, supply-chain).
- You have enough **training data** (years of daily returns × N assets).

For small universes (<50) or very short histories, vanilla XGBoost on engineered cross-sectional features usually wins. For large panels with clear structure, GNNs can add 0.05-0.15 IC over the GBM baseline.

The frontier-track use: combine GNN-derived node embeddings with a downstream GBM. Best of both worlds — GNN for relational structure, GBM for non-linear interactions.

Continue to **[Diffusion-augmented scenario backtesting](06-diffusion-scenarios.md)**.
