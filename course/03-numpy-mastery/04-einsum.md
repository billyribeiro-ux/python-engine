# `einsum` and tensor contractions

`np.einsum` is a one-function generalisation of matrix products, dot products, traces, outer products, transpositions, and almost every other "shuffle axes and sum some of them" operation. Once you learn its mini-language, you stop reaching for `np.dot`, `np.matmul`, `np.tensordot`, `np.outer`, and `np.inner` — `einsum` does them all, more clearly, and often faster.

## The mini-language, in one example

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])      # shape (2, 2)
B = np.array([[5, 6], [7, 8]])      # shape (2, 2)

# Matrix product the normal way
C1 = A @ B                          # array([[19, 22], [43, 50]])

# Same thing with einsum
C2 = np.einsum("ij,jk->ik", A, B)
np.allclose(C1, C2)                  # True
```

The string `"ij,jk->ik"` is the *Einstein summation* notation:

- `A` has axes labelled `i, j` (rows, cols).
- `B` has axes labelled `j, k`.
- The arrow `->` says: the output has axes `i, k`.
- The axis label `j` appears on both inputs but not on the output → **sum over it**.

That's the whole rule:

> *Labels that appear on inputs but not the output are summed over. Labels that appear on both inputs are matched (broadcast / dot). Labels on a single input are kept.*

## A cheat sheet

| Operation | `einsum` |
|---|---|
| Transpose | `"ij->ji"` |
| Sum all elements | `"ij->"` |
| Sum each row | `"ij->i"` |
| Sum each column | `"ij->j"` |
| Vector dot | `"i,i->"` |
| Outer product | `"i,j->ij"` |
| Matrix-vector | `"ij,j->i"` |
| Matrix-matrix | `"ij,jk->ik"` |
| Batched matrix-matrix | `"bij,bjk->bik"` |
| Diagonal | `"ii->i"` |
| Trace | `"ii->"` |
| Hadamard (element-wise) | `"ij,ij->ij"` |

Internalise this table and you can do anything.

## Why you'd actually use it

### 1. Self-documenting axis bookkeeping

```python
# Compare:
np.tensordot(A, B, axes=([1, 2], [0, 1]))
# vs:
np.einsum("ijk,jkl->il", A, B)
```

The second is obvious about what's happening. The first is a puzzle.

### 2. Batched operations

```python
# 100 batches of (3x4 @ 4x5) matrix products
A = np.random.randn(100, 3, 4)
B = np.random.randn(100, 4, 5)

# einsum way: read it like a sentence
C = np.einsum("bij,bjk->bik", A, B)
# C.shape == (100, 3, 5)
```

`np.matmul` does the same thing (`A @ B`), but einsum scales to *any* dimensional layout. For 4D tensors with named axes (e.g. attention layers in transformers), it's the only sane way.

### 3. Implicit-sum tricks that have no other one-liner

The **squared Euclidean distance matrix** between N points:

```python
X = np.random.randn(N, D)
sq_norms = np.einsum("ij,ij->i", X, X)       # ||x_i||^2 per row
D2 = sq_norms[:, None] + sq_norms[None, :] - 2 * np.einsum("ik,jk->ij", X, X)
# D2[i, j] == ||x_i - x_j||^2
```

Three lines, no allocation of an (N, N, D) intermediate. Compare to the naive broadcast version `((X[:, None] - X[None, :]) ** 2).sum(-1)` which allocates that whole (N, N, D) cube — fine for N=200, painful for N=20,000.

### 4. Returns to portfolio P&L

```python
# returns shape (T, N): T days, N symbols
# weights shape (T, N): your time-varying portfolio
pnl = np.einsum("ti,ti->t", returns, weights)   # daily portfolio return
```

`np.einsum` here is the cleanest expression of "elementwise multiply, then sum over symbols". You could also write `(returns * weights).sum(axis=1)` — same thing, slightly less compact, allocates a temp matrix.

### 5. Covariance matrices

```python
# X centred, shape (T, N). Covariance matrix:
cov = np.einsum("ti,tj->ij", X, X) / (X.shape[0] - 1)
```

Same as `X.T @ X / (T-1)`, but for higher-rank generalisations (4D, attention) it's the same syntax.

## Performance

Modern `einsum` is competitive with `np.matmul` / BLAS for the common cases (because it dispatches to them internally). For complex expressions, `np.einsum_path` or the `optimize="optimal"` flag finds the best contraction order:

```python
C = np.einsum("ij,jk,kl,lm->im", A, B, C, D, optimize="optimal")
```

The cost of `((A @ B) @ C) @ D` vs `A @ (B @ (C @ D))` can differ by orders of magnitude when shapes vary. `einsum` figures out the cheapest order.

## When NOT to reach for einsum

- Simple 2D matrix product → `A @ B`. Faster to read.
- Element-wise stuff → just write `A * B`.
- One-axis sum → `A.sum(axis=...)`. Clearer.

Use `einsum` when the structure is **non-trivial** or the **named axes help readability**. Don't einsum-pilled your whole codebase — taste matters.

## A real-world example: covariance shrinkage

A Ledoit-Wolf-flavoured covariance estimator on returns matrix $X$ of shape $(T, N)$:

$$
\Sigma = \alpha \cdot \operatorname{diag}(\hat\sigma^2) + (1-\alpha) \cdot \hat\Sigma_{\text{sample}}
$$

```python
def shrink_cov(X: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    T, N = X.shape
    Xc = X - X.mean(axis=0)
    sample = np.einsum("ti,tj->ij", Xc, Xc) / (T - 1)       # sample covariance
    diag = np.diag(np.einsum("ti,ti->i", Xc, Xc) / (T - 1)) # diagonal target
    return alpha * diag + (1 - alpha) * sample
```

Two einsums and a diag. No explicit loops. Reads like the formula.

## A trap

`einsum` is happy to silently broadcast over repeated labels in the same input — and that's almost never what you want:

```python
A = np.array([[1, 2], [3, 4]])
np.einsum("ii", A)       # trace: 1 + 4 = 5
np.einsum("ii->i", A)    # diagonal: [1, 4]
```

The `"ii"` in the first form is the **trace shortcut**. If you meant a plain elementwise op, use distinct letters. Double-check your subscript strings against the cheat sheet when results surprise you.

Continue to **[Strided rolling windows](05-strided-windows.md)**.
