# Module 14 — Options Foundations

If you've made it this far without trading options, this is where the course branches. Options are not "leveraged stock" — they're a non-linear payoff that depends on the *distribution* of the underlying, not just its mean. Pricing one requires a model. Hedging one requires Greeks. Trading options well requires both, plus respect for the failure modes the equity-only crowd never has to think about.

This module is the working foundation. The `engine.options` package — built incrementally in this and Module 15 — gives you production-quality Black-Scholes pricing, Greeks (including the cross-Greeks), implied volatility, American exercise via binomial trees, and the stochastic-vol pricing methods. Everything is testable, vectorised, and used in the strategy modules that follow.

Pages:

1. **[Black-Scholes properly](01-black-scholes.md)** — the derivation in five lines and the working code in `engine.options.price`.
2. **[All the Greeks: first, second, and the cross ones you've never used](02-greeks.md)** — delta, gamma, vega, theta, rho, vanna, charm, vomma.
3. **[Implied volatility — robust solvers](03-implied-vol.md)** — Brent over Newton; the deep ITM/OTM trap.
4. **[American options — binomial trees and LSMC](04-american.md)** — early-exercise pricing for equity options.
5. **[Put-call parity and synthetic positions](05-parity.md)** — what you can construct without owning the underlying.

Start with **[Black-Scholes properly](01-black-scholes.md)**.
