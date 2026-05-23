# How to read this course

There is more material here than any sensible person reads in a single sitting. Here is how to get the most out of it.

## Three reading passes

### Pass 1 — skim everything, run nothing

Open every module. Read the **bold sentences**, the diagrams, and the admonitions like the one below. Skip the code. The goal is to build a **mental map** so you know what lives where when you need to come back.

!!! tip "Why the skim matters"
    The frontier track in particular has methods that *only* make sense once you know they exist. You will not invent persistent homology on your own at 2am. But you might remember "there was a chapter on TDA fragility" and find it in two minutes.

A first-pass skim of all 21 modules should take you a focused afternoon, maybe two.

### Pass 2 — read for understanding, type along

On the second pass, you read sequentially and **type the code**, not paste it. Yes, the copy button is right there. Use it for the imports. Type the algorithmic body. Typing is how you notice you don't actually understand a line — pasting hides that from you.

Run every code block. Each one is self-contained: copy the imports at the top of the chapter and any block in that chapter will run.

### Pass 3 — reference, on demand

After pass 2 the course becomes a search-indexed reference. Press <kbd>/</kbd>, type "Kalman" or "GEX" or "purged k-fold", and you'll land on the page you need in under a second.

## Admonitions you'll see

The course uses a few standard call-out blocks. Each one means something specific.

!!! note "Note"
    Background context or a related fact. Skippable on the first pass.

!!! tip "Tip"
    A non-obvious shortcut or insight. Worth reading.

!!! warning "Warning"
    A genuine footgun. Most of these come from real production incidents.

!!! danger "Danger"
    A footgun that costs money. Read these slowly.

!!! example "Example"
    A worked numerical example with output. Run it.

!!! success "Production"
    This pattern ships at real funds. Use it.

!!! abstract "Frontier"
    This is on the frontier track — exotic and powerful, but not the right first tool. Read it after you have the basics.

## Math and notation

Math is rendered with [MathJax](https://www.mathjax.org/). When a result matters, the derivation is written out; when it doesn't, the formula is stated and cited. You don't need to be a mathematician to follow the course — every formula has a code block underneath that does the same thing in NumPy or PyTorch.

Inline math uses single dollar signs: $\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2$.

Display math uses double dollars:

$$
\hat{x}_{t\mid t} = \hat{x}_{t\mid t-1} + K_t \left( z_t - H \hat{x}_{t\mid t-1} \right)
$$

If a chapter has math you don't follow, **skip the derivation and read the code**. The code is the source of truth in this course.

## "Plain English at L7+" — what that means

A common mistake in technical writing is treating "rigorous" and "complicated" as synonyms. They aren't. This course is written under a single rule:

> **Every sentence aims to be the simplest sentence that is still true.**

That means:

- No jargon without a definition the first time it appears.
- No hand-waving the parts that matter, even when they're hard.
- No theatrical complexity — if a one-line function does the job, the chapter shows the one line.

If you find a passage that breaks the rule, that's a bug. Open an issue.

Continue to **[Environment setup](setup.md)**.
