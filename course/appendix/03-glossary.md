# Glossary

Definitions for the terms used across the course. Where a term has both a strict technical meaning and a more casual usage, both are noted.

**Adverse selection**: when your counterparty trades against you systematically because they have information you don't. The dominant cost in HFT.

**ADV (Average Daily Volume)**: typical daily volume traded in a symbol. The standard liquidity proxy.

**Alpha**: the risk-adjusted excess return of a strategy after benchmark exposure. In a factor model, the regression intercept.

**Almgren-Chriss**: the canonical optimal-execution scheduling framework. Module 19 chapter 5.

**Avellaneda-Stoikov**: the canonical optimal-quoting model for market making. Module 19 chapter 4.

**Backtest**: a simulation of how a strategy would have performed historically.

**BBO (Best Bid and Offer)**: the highest bid and lowest ask currently in the market.

**Beta**: the regression slope of an asset's returns against a benchmark's returns.

**Black-Scholes-Merton**: the closed-form European-option pricing formula. Module 14 chapter 1.

**CVaR (Conditional Value-at-Risk, Expected Shortfall)**: the average loss conditional on exceeding VaR. Coherent risk measure.

**Cointegration**: two non-stationary series whose linear combination is stationary. The basis of pairs trading.

**Conformal prediction**: distribution-free framework for prediction intervals with finite-sample coverage guarantees. Module 10 chapter 5.

**Convexity**: in options, the curvature of the price-vs-underlying curve. Long convexity = gamma > 0.

**Cumulative Distribution Function (CDF)**: $F(x) = P(X \leq x)$.

**Cython**: language that compiles a typed Python superset to C. Used in pandas' internals.

**Decile / Quintile**: 10% / 20% slices of a distribution.

**Deflated Sharpe Ratio (DSR)**: Sharpe adjusted for multiple-testing and sample-size effects. Module 9 chapter 5.

**Drawdown**: peak-to-trough loss expressed as a fraction.

**EVT (Extreme Value Theory)**: framework for modeling the tail of a distribution. Module 6 chapter 5.

**Fama-French model**: 3-factor (market, size, value) and 5-factor (+ profitability, investment) asset-pricing models.

**Fractional differencing**: a generalisation of first-differencing that preserves long-memory structure. Module 7 chapter 1 / Module 10 chapter 2.

**Gamma (Γ)**: second derivative of option price wrt underlying. Captures convexity.

**GARCH**: autoregressive conditional heteroskedasticity model for volatility. Module 7 chapter 2.

**GBM (Geometric Brownian Motion)**: $dS = \mu S dt + \sigma S dW$. The Black-Scholes underlying.

**GBM (Gradient Boosted Machine)**: ML model. XGBoost, LightGBM, CatBoost.

**Greeks**: partial derivatives of option price wrt its inputs.

**HRP (Hierarchical Risk Parity)**: portfolio construction via clustering + recursive bisection. Module 8 chapter 5 / Module 20 chapter 5.

**Idempotency key**: client-side unique identifier on an order so retries don't double-trade. Module 21 chapter 3.

**Implied volatility (IV)**: the σ that makes Black-Scholes match the observed market price.

**Implementation shortfall**: cost vs the arrival price (when the trade decision was made).

**Information coefficient (IC)**: rank correlation between predicted and realised values.

**Kalman filter**: optimal recursive estimator for linear-Gaussian systems. Module 7 chapter 3.

**Kelly criterion**: $f^* = \mu / \sigma^2$. Optimal bet fraction for log-growth. Module 8 chapter 6.

**Kyle's lambda**: marginal price impact per unit volume. Module 19 chapter 2.

**Limit Order Book (LOB)**: full list of resting bids and asks.

**Long-Short**: portfolio holding some longs and some shorts; usually dollar-neutral.

**Market impact**: the price move caused by your own trade.

**Mean reversion**: the tendency of a series to return to its average.

**Microstructure**: the study of order-book dynamics, trades, and quotes at high frequency.

**Momentum**: the tendency of recent winners to continue winning.

**NumPy**: the foundational Python numerical array library.

**OFI (Order Flow Imbalance)**: signed change in top-of-book size; predictive of next mid-price move.

**Pandas**: dataframe library on top of NumPy.

**Parquet**: columnar on-disk storage format. The standard for analytical data.

**PIT (Point-In-Time)**: data as it was known at a specific historical date. Module 5 chapter 3.

**Polars**: Rust-backed dataframe library, alternative to pandas.

**Pre-averaging**: noise-correction technique for tick-level realised vol. Module 19 chapter 3.

**Purged k-fold**: time-series CV that drops samples whose label horizon overlaps the test fold. Module 9 chapter 4.

**Quintile**: 20% slice of a distribution.

**Realised volatility**: vol computed from observed price moves (not implied).

**Reservation price**: in market making, the spot offset the MM would prefer to quote around.

**SABR**: stochastic-vol model used in rates / FX. Module 15 chapter 2.

**Sharpe ratio**: annualised return / annualised vol.

**Slippage**: difference between intended and realised execution price.

**Spread**: best ask − best bid.

**Stationarity**: a series whose statistical properties don't change over time. Module 7 chapter 1.

**Stochastic volatility (SV)**: vol that itself follows a stochastic process. Heston is the canonical SV model.

**SVI (Stochastic Volatility Inspired)**: parametric smile fitting formula. Module 15 chapter 1.

**Tenor**: maturity of an option or bond.

**Term structure**: how a quantity (yield, IV) varies with maturity.

**Tick**: a single quote or trade update.

**TWAP / VWAP**: Time-Weighted / Volume-Weighted Average Price. Standard execution benchmarks.

**Universe**: the set of symbols the strategy considers.

**VaR (Value-at-Risk)**: the loss that exceeds with probability α.

**Vega (ν)**: option price sensitivity to vol.

**VIX**: market-implied 30-day expected SPX volatility.

**Vol of vol**: variance of variance. Captured by Heston's $\sigma_v$.

**Volatility regime**: a multi-day period of consistent vol level.

**VPIN**: Volume-synchronised Probability of Informed Trading. Module 19 chapter 2.

**Walk-forward**: backtest pattern that retrains the model periodically using only past data.

**XGBoost / LightGBM / CatBoost**: gradient-boosted tree libraries.

**Z-score**: (value − mean) / std. Standardised distance from average.

Continue to **[Common mistakes you'll make](04-common-mistakes.md)**.
