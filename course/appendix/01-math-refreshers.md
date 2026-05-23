# Math refreshers

You don't need to be a mathematician to use this course, but a working command of these primitives makes everything easier. This page is the cheat-sheet, not a textbook.

## Linear algebra

### Vectors and matrices

A vector $\mathbf{v} \in \mathbb{R}^n$ is an ordered list of $n$ numbers. A matrix $A \in \mathbb{R}^{m \times n}$ is a rectangular array of numbers with $m$ rows and $n$ columns.

Matrix multiplication: $(AB)_{ij} = \sum_k A_{ik} B_{kj}$. Requires $A$'s column count to equal $B$'s row count.

### Inner product, outer product

Inner product (dot product): $\mathbf{u} \cdot \mathbf{v} = \sum_i u_i v_i$. A scalar.

Outer product: $(\mathbf{u} \mathbf{v}^\top)_{ij} = u_i v_j$. A matrix.

### Norms

$L^2$ norm: $\|\mathbf{v}\|_2 = \sqrt{\sum_i v_i^2}$. The Euclidean length.
$L^1$ norm: $\|\mathbf{v}\|_1 = \sum_i |v_i|$. Used in LASSO regression.
$L^\infty$ norm: $\|\mathbf{v}\|_\infty = \max_i |v_i|$.

### Eigenvalues and eigenvectors

For a square matrix $A$, an eigenvector $\mathbf{v}$ and eigenvalue $\lambda$ satisfy $A\mathbf{v} = \lambda \mathbf{v}$.

For a symmetric matrix (like a covariance), all eigenvalues are real and eigenvectors are orthogonal. This is what PCA uses.

### Singular value decomposition (SVD)

Any matrix $A$ decomposes as $A = U \Sigma V^\top$ where $U$ and $V$ are orthogonal and $\Sigma$ is diagonal with non-negative entries (the singular values).

The largest singular value gives the matrix's spectral norm. SVD is the basis of PCA, low-rank approximations, and pseudo-inverses.

### Positive (semi-)definite matrices

A matrix $M$ is **positive semi-definite** if $\mathbf{x}^\top M \mathbf{x} \geq 0$ for all $\mathbf{x}$. **Positive definite** if strictly $>$ for $\mathbf{x} \neq \mathbf{0}$.

Covariance matrices are PSD; correlation matrices are PSD (and have 1s on the diagonal).

### Computing in NumPy

```python
import numpy as np

# Matrix multiplication
C = A @ B

# Solve Ax = b
x = np.linalg.solve(A, b)               # better than inv(A) @ b

# Eigendecomposition (symmetric)
eigvals, eigvecs = np.linalg.eigh(A)    # sorted ascending

# SVD
U, S, Vt = np.linalg.svd(A)

# Norms
np.linalg.norm(v, ord=2)
```

## Probability

### Random variables

A random variable $X$ is a function from outcomes to numbers. Characterised by its **distribution**.

For a continuous RV with PDF $f$:
- Mean: $\mathbb{E}[X] = \int x f(x) dx$.
- Variance: $\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]$.
- Cov: $\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]$.

### Common distributions

- **Normal $\mathcal{N}(\mu, \sigma^2)$**: bell curve; $\mathbb{E} = \mu$, $\text{Var} = \sigma^2$. Foundation of Black-Scholes (which assumes log-prices are normal).
- **Student-t**: fat-tailed alternative to normal. Used for residuals in GARCH (Module 7).
- **Lognormal**: $X$ is lognormal iff $\log X$ is normal. Used for prices.
- **Generalised Pareto (GPD)**: heavy-tailed; used in EVT (Module 6).
- **Poisson**: counts of independent events. Hawkes is its self-exciting cousin.

### Conditional probability and Bayes' rule

$$P(A | B) = \frac{P(A \cap B)}{P(B)} = \frac{P(B | A) P(A)}{P(B)}$$

The basis of Bayesian inference (Module 6) and Hidden Markov Models.

### Central limit theorem

Sum of many independent, finite-variance RVs is approximately normal. The reason normal-based intervals work for many statistics — but **does not apply** to fat-tailed financial returns directly.

## Stochastic calculus essentials

### Brownian motion

$W_t$ is standard Brownian motion if:
- $W_0 = 0$.
- $W_t - W_s \sim \mathcal{N}(0, t - s)$ for $s < t$.
- $W$ has independent increments.

The "infinitely-jagged" continuous random walk.

### Itô's lemma

For a function $f(t, X_t)$ where $dX_t = \mu_t dt + \sigma_t dW_t$:

$$df = \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial X} dX + \frac{1}{2} \frac{\partial^2 f}{\partial X^2} \sigma_t^2 dt$$

The "$\frac{1}{2} \sigma^2$" correction is the key non-classical piece. It's what gives Black-Scholes its specific form.

### Geometric Brownian motion (GBM)

$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

Solution: $S_t = S_0 \exp((\mu - \frac{1}{2}\sigma^2) t + \sigma W_t)$.

Black-Scholes' underlying model.

### Ornstein-Uhlenbeck (OU)

$$dX_t = \theta (\mu - X_t) dt + \sigma dW_t$$

Mean-reverting. The canonical model for stationary spreads (Module 8 chapter 3).

### Risk-neutral measure

Under the **risk-neutral measure $\mathbb{Q}$**, all assets earn the risk-free rate. Option prices are computed as discounted expected payoffs under $\mathbb{Q}$.

For Black-Scholes: $dS_t = r S_t dt + \sigma S_t dW^\mathbb{Q}_t$. The drift is $r$, not $\mu$. The volatility $\sigma$ is the same — Girsanov's theorem.

### Heston (stochastic vol)

$$dS_t = \mu S_t dt + \sqrt{v_t} S_t dW^1_t$$
$$dv_t = \kappa (\theta - v_t) dt + \sigma_v \sqrt{v_t} dW^2_t$$

Two coupled SDEs. The variance $v_t$ is itself stochastic. Module 15.

## Notation conventions used in the course

- $S$ — underlying asset price.
- $K$ — strike price.
- $T$ — expiry (in years).
- $t$ — time elapsed (in years).
- $r$ — risk-free interest rate (continuous).
- $q$ — dividend yield (continuous).
- $\sigma$ — volatility (annualised).
- $\mu$ — drift / expected return.
- $\Delta, \Gamma, \nu, \Theta, \rho$ — option Greeks.
- $W_t$ — standard Brownian motion.

Continue to **[Reading list](02-reading-list.md)**.
