# Postmortems and not over-tuning

Things will go wrong. A strategy that worked in backtest will underperform live. A bug will produce an unintended trade. An external event (data feed outage, broker API change) will surface a fragility you didn't know about. Every shop encounters all of these. The difference between funds that recover and funds that compound their problems is how they handle each incident.

This chapter is the discipline.

## What a good postmortem looks like

A postmortem is a written document covering:

1. **What happened** — the factual sequence, with timestamps.
2. **Impact** — P&L, position, reputational. Quantified.
3. **Root cause** — the underlying reason, not just the proximate trigger.
4. **Action items** — what changes prevent recurrence. With owners and dates.
5. **Lessons learned** — what general principles apply beyond this specific incident.

Blameless. The point is to make the system better, not to assign fault. Engineers who fear blame hide bugs; engineers who can speak openly help fix them.

## A template

```markdown
# Postmortem: SPY Pairs Strategy Loss, 2024-11-15

## Summary
On 2024-11-15, the SPY-IVV pairs strategy realised -$45,000 (vs typical day's P&L of ±$2,000).
Trading was halted by the daily-drawdown kill switch at 14:32 EST.

## Timeline (all times EST)
- 09:30 — Market open. Strategy positions at planned levels.
- 13:15 — IVV spread widened to 5σ on a single print (likely IVV erroneous quote).
- 13:16 — Strategy entered a 2× normal position, expecting reversion.
- 13:20 — Spread continued widening; strategy entered another 2× position.
- 14:32 — Daily loss exceeded $30k; kill switch fired.
- 14:33 — Watchdog closed positions at market.
- 14:35 — Investigation began.

## Impact
- Realised loss: $45,000 (2.8% of strategy capital).
- 6 unexpected trades.
- No reputational impact (no investor visibility).

## Root cause
The strategy's hard-stop logic was z-score-based, but used a *static* z-score threshold (4σ).
The IVV print's erroneous value caused the rolling-window variance to update, which
artificially inflated the threshold. By the time the threshold would have triggered,
the loss was already large.

## Contributing factors
1. No filter for "anomalous tick" before updating the rolling-window stats.
2. The strategy doubled down on the position rather than capping.
3. The daily-drawdown kill switch fired correctly but at $30k — too lax for this strategy.

## Action items
- [ ] Add anomalous-tick filter to all spread-based strategies. Owner: Alice. Due: 2024-11-22.
- [ ] Cap intraday position growth to 1× nominal (no doubling). Owner: Bob. Due: 2024-11-29.
- [ ] Reduce daily-drawdown kill switch to $15k for this strategy. Owner: Bob. Due: 2024-11-20.

## Lessons learned
- Strategies that "double down" on adverse moves need particular care.
- Hard stops on path-dependent statistics (like rolling z-scores) can fail in surprising ways.
- Kill-switch thresholds should be strategy-specific, not portfolio-uniform.
```

Write this for every incident, even small ones. Read all past postmortems before designing a new strategy.

## The "don't over-tune" discipline

After a loss, the strongest temptation is to **change something** — anything — to feel like you're fixing it. This is usually wrong.

Reasons to NOT immediately re-tune:

1. **One observation is one observation.** A single bad day doesn't invalidate the strategy. The expected variance is *huge*.
2. **Re-tuning on the bad day biases the strategy** toward avoiding *that specific kind of bad day* — which is unlikely to repeat.
3. **The cost of being wrong matters more than the cost of waiting.** A worse strategy actively trading is much worse than a fine strategy on hold.

The right protocol after a loss:

1. **Pause** — kill switch if not already fired.
2. **Investigate** — root cause analysis (the postmortem above).
3. **Decide** — was this a bug, a regime change, or a normal-distribution tail?
4. **Act** — only after the decision, and only on the root cause.

For "normal-distribution tail" outcomes: resume trading without changes.

For "regime change": consider pausing or reducing size; re-evaluate after the regime stabilises.

For "bug": fix the specific bug; don't refactor the strategy.

## A specific anti-pattern

```
Day 1: -3% loss. "Maybe I should reduce max_drawdown to 5%."
Day 2: +1% gain.
Day 5: -1% loss. "Maybe I should add a regime filter."
Day 8: -2% loss. "Maybe I should..."
```

Each tweak biases the strategy toward avoiding the kind of loss that just happened. The strategy that emerges fits the recent N days perfectly and will underperform on day N+1. **Strategy parameters change at most quarterly**, not after every bad day.

## When to actually re-tune

Re-tuning is appropriate when:

- A **structural change** has happened in the market (e.g., regulatory regime change, major market participant exit). Not "vol was higher than usual."
- A **CV-based reassessment** with a year of out-of-sample data confirms the original parameters are sub-optimal.
- A **bug fix** reveals the original parameters were based on incorrect numbers.

Each of these is rare. If you're re-tuning more often than quarterly, you're over-tuning.

## Capacity and crowding

A separate failure mode: the strategy is fine but the *market* has changed. If alpha decays because too many other people run the strategy, no re-tuning of *your* strategy will fix it.

Indicators:

- Backtest Sharpe degrades over years even after re-fitting.
- Trade fills cluster at predictable times (you're not the only one trading at the close).
- Conformal intervals widen — the model is less confident.

The right response: **retire the strategy**, not "make it better." Free up the capital for something new.

## Investor communication

When something goes wrong, communicate proactively:

- **Within hours**: send investors a one-paragraph summary of the incident.
- **Within 1-2 days**: send a fuller postmortem (sanitised, no proprietary details).
- **Within a week**: hold a call if requested.

Investors will accept "we had a bad day; here's what happened; here's what we changed." They will not accept silence or surprise next month's statement.

## The end-of-day routine

Every day after the close:

1. Check P&L vs backtest expectation.
2. Reconcile positions broker vs local.
3. Read all warning-level logs.
4. Run a slippage report (paper chapter's reconciliation).
5. Update the strategy journal — what happened, what's expected tomorrow.

This routine is the difference between systems that drift into trouble and systems that catch problems on day 1.

## Quarterly strategy review

Once a quarter:

- Compute the strategy's realised vs expected Sharpe.
- Compute the realised vs expected drawdown distribution.
- Apply DSR / PBO (Module 9 chapter 5) to ALL strategies in the book.
- Identify strategies whose DSR has degraded → candidates for retirement.
- Identify strategies whose IC has held up → candidates for sizing up.

## Pitfalls

!!! warning "Blame culture"
    A culture that blames individuals for incidents creates hidden bugs. Blameless postmortems are non-negotiable.

!!! warning "Postmortem without action items"
    A postmortem without specific, dated, owned action items is just venting. Each item must be a commitment.

!!! warning "Over-active strategy lifecycle**
    Strategies that get re-tuned every month never get a chance to prove themselves. Pick a parameter set; commit; revisit quarterly.

!!! warning "Letting capacity quietly erode"
    Strategies decay. Track Sharpe over rolling windows; have a written "retire when" criterion.

## Bottom line

For production trading:

- **Postmortem every incident**, blamelessly, in writing, with action items.
- **Don't re-tune after a single bad day** — root-cause first.
- **Quarterly strategy reviews** with explicit retire-when criteria.
- **Daily routines** that catch drift early.
- **Investor communication** that's proactive and honest.

## End of Module 21 and the course

You now have, in roughly this order:

- Python foundations and hacks (Modules 1-2).
- NumPy / pandas / polars (Modules 3-4).
- Data engineering for markets (Module 5).
- Statistics and time series (Modules 6-7).
- Classical quant + backtesting + ML foundations (Modules 8-10).
- Modern ML, deep learning, RL (Modules 11-13).
- Options machinery (Modules 14-15).
- Production and frontier strategies (Modules 16-17).
- Scanners (Module 18).
- Execution and microstructure (Module 19).
- Risk and portfolio (Module 20).
- Deployment (Module 21).

What's left is the [**Appendix**](../appendix/index.md) — math refreshers, reading list, glossary, and the "common mistakes you'll make" list.
