# Hypothesis testing and the p-hacking trap

The textbook setup: pick a null hypothesis, compute a test statistic, reject if p<0.05, publish. The problem with applying this to trading: you're not running *one* test. You're running thousands. The framework that controls error rates for one test makes essentially no guarantees for thousands.

This is the most important chapter in Module 6. Internalising it is the difference between "researcher who finds 'edges' all the time" (problematic) and "researcher who finds genuine edges that work in live trading" (the goal).

## The setup

You backtest 1,000 random strategies. By chance, if each has a true expected Sharpe of 0, you'd expect about 50 of them to have a backtested Sharpe whose p-value is below 0.05. You pick the best one. Its in-sample Sharpe is 1.8. You ship it.

Live, it performs at Sharpe 0.

This is not bad luck. It is the **multiple-testing problem**, and it is the dominant reason backtests don't replicate.

## A useful framing

The Sharpe ratio you observe in a backtest, $\hat S$, is a random variable. Even when the true Sharpe is 0, $\hat S$ has a distribution — its standard deviation under the null is approximately:

$$
\sigma(\hat S) \approx \sqrt{\frac{1 + \tfrac{1}{2}\hat S^2}{T - 1}}
$$

For a 5-year daily backtest ($T \approx 1{,}260$), the standard deviation is roughly 0.028. So a backtest Sharpe of $\hat S = 0.5$ is *barely* 1.5 standard deviations above zero — for a single test, marginally interesting; for one of a thousand tests, expected to happen by chance.

## The deflated Sharpe ratio

Bailey & Lopez de Prado (2014) formalised the fix: the **deflated Sharpe ratio (DSR)** adjusts an observed Sharpe for the number of trials and the variance of the trial Sharpes. The intuition: if you tried 1,000 things, the best one's Sharpe should look much better than the "typical Sharpe under the null" to count as real.

The full formula is in the paper. A working sketch:

```python
import numpy as np
from scipy.stats import norm

def deflated_sharpe(sharpes: np.ndarray, T: int) -> float:
    """Probability the maximum observed Sharpe across N trials is real."""
    N = len(sharpes)
    sharpes = np.asarray(sharpes)
    s_max = sharpes.max()
    # Expected max of N independent N(0, sd^2) draws via Mises (1936) approx
    sd = sharpes.std(ddof=1)
    e_max = sd * ((1 - np.euler_gamma) * norm.ppf(1 - 1/N)
                  + np.euler_gamma * norm.ppf(1 - 1/(N * np.e)))
    # z-score the observed against the null
    se = np.sqrt((1 - sharpes.mean() * s_max + 0.5 * s_max ** 2) / (T - 1))
    z = (s_max - e_max) / se
    return float(norm.cdf(z))      # probability of "real"
```

Run a thousand random strategies, get their backtest Sharpes, feed them in with your champion's Sharpe. The DSR tells you: given the noise level you've observed across trials, how surprising is the best one?

If DSR < 0.9, you don't have a real edge — you have selection.

## The Bonferroni and Holm corrections

For controlling the family-wise error rate (FWER) across $K$ tests:

- **Bonferroni** — reject only when $p < \alpha / K$. Conservative.
- **Holm** — sort p-values, reject the smallest if $p_{(1)} < \alpha/K$, the next if $p_{(2)} < \alpha/(K-1)$, ... Slightly more powerful, same FWER guarantee.

In code:

```python
from scipy.stats import false_discovery_control

# Bonferroni
adjusted = pvalues * K
reject = adjusted < alpha

# Benjamini-Hochberg (controls FDR, less stringent)
adjusted = false_discovery_control(pvalues, method="bh")
```

For trading research, **FDR (Benjamini-Hochberg) is usually the right control level**. You can tolerate *some* false positives if you re-verify out of sample; you can't tolerate the FWER's mass rejection of borderline-real signals.

## A practical rule of thumb

The honest research workflow:

1. **Hold out** at least 30% of your time series as a final test set. Do not look at it.
2. **Develop and tune** on the rest. Try as many ideas as you want.
3. **Promote to the holdout** only when you've decided on the final model.
4. **Promote to paper trading** for at least 30 trading days.
5. **Promote to live** only after paper-trading matches your expectations.

If your strategy doesn't survive each promotion, it wasn't real. The DSR machinery formalises step 3; the *discipline* of refusing to peek at the holdout is what actually saves you.

## p-values vs effect sizes

A second under-appreciated trap: even a "real" small effect can be detected with a giant sample. p<0.001 means "this is unlikely under the null," not "this is economically meaningful."

For a trading strategy, **report the effect size**: annualised return, annualised Sharpe, max drawdown, hit rate. A 0.4 Sharpe with p=0.001 may be statistically real and economically dead after transaction costs.

## Power: the opposite blind spot

A high p-value (> 0.05) doesn't mean "no effect" — it means "this study didn't detect one." With short time series (1 year of daily data), even a real Sharpe-1 strategy fails the standard test about half the time:

```python
import numpy as np
from scipy.stats import norm

def power_for_sharpe(true_sharpe: float, T: int, alpha: float = 0.05) -> float:
    se = 1 / np.sqrt(T)
    z = norm.ppf(1 - alpha)
    return float(norm.sf(z - true_sharpe / se))

print(f"{power_for_sharpe(1.0, 252):.0%}")    # ~46% — flip of a coin to detect a Sharpe-1!
print(f"{power_for_sharpe(1.0, 252*3):.0%}")  # ~83% — three years is much better
print(f"{power_for_sharpe(1.0, 252*5):.0%}")  # ~96%
```

**Demand long backtests**. A 1-year backtest of even a great strategy is hard to distinguish from noise.

## Stationarity test mechanics (preview of Module 7)

Most statistical tests assume stationarity. Returns are roughly stationary; prices are not. ADF and KPSS tests are how you check:

```python
from statsmodels.tsa.stattools import adfuller, kpss

p_adf = adfuller(returns)[1]      # H0: series has a unit root (non-stationary)
p_kpss = kpss(returns, regression="c")[1]   # H0: series is stationary

# Roughly: reject ADF, fail to reject KPSS → confidently stationary
```

We come back to this in Module 7 with the full battery.

## A specific anti-pattern: in-sample parameter sweep

```python
best_sharpe, best_window = -np.inf, None
for w in range(5, 60):
    sharpe = backtest(window=w).sharpe()
    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_window = w
print(f"Optimal window: {best_window}, in-sample Sharpe {best_sharpe:.2f}")
```

You searched 55 windows. The best is — by construction — better than the average. The correct interpretation: "the in-sample Sharpe of the winner gives you essentially no information about its out-of-sample Sharpe."

The fix is **walk-forward cross-validation** (Module 9): you fit on history, freeze, test on the next period. Repeat. The Sharpes you collect are out-of-sample.

## Bayesian shrinkage of strategy means

Even after you've controlled for multiple testing, there's one more trap: the strategies that did best in your sample are the ones whose true means you've *over-estimated*. The honest correction is shrinkage:

$$
\hat\mu_{\text{posterior}} = \lambda \cdot \hat\mu_{\text{prior}} + (1-\lambda) \cdot \hat\mu_{\text{sample}}
$$

with $\lambda$ a function of the sample size and the prior tightness. The next chapter (Bayesian) makes this concrete.

## Bottom line

The discipline:

1. **Time-series CV, no in-sample peeking.**
2. **Use DSR (or at least Bonferroni / FDR) when comparing many candidates.**
3. **Report effect size, not just p-values.**
4. **Require at least 3–5 years of out-of-sample backtest** for daily strategies.
5. **Paper-trade before going live, always.**

The cost of this discipline is *not* shipping the great-looking-but-fake strategies. That is also its benefit.

Continue to **[The bootstrap, properly](02-bootstrap.md)**.
