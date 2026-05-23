# Module 12 — Deep Learning for Time Series

For low-SNR daily-bar prediction, gradient boosting usually wins (Module 11). But there are domains where deep nets genuinely outperform: very-long sequence modelling, multi-task learning across thousands of related series, raw text/audio/image alternative data, and continuous-time models (neural ODEs/SDEs). This module is the working tour of the modern deep architectures, with a focus on what actually moves the needle for trading.

Pages:

1. **[When deep learning beats gradient boosting](01-when-dl-wins.md)** — the honest decision framework.
2. **[TCN, N-BEATS, N-HiTS](02-tcn-nbeats.md)** — the workhorse forecasting architectures.
3. **[PatchTST, iTransformer, TimesNet — the transformer family for time series](03-transformers-ts.md)** — what's actually new since 2022.
4. **[Mamba and state-space models](04-mamba.md)** — the post-transformer landscape.
5. **[Neural ODEs and SDEs](05-neural-ode-sde.md)** — continuous-time models for irregular data.
6. **[Diffusion models for synthetic market paths](06-diffusion.md)** — generative models for stress testing and data augmentation.

Start with **[When deep learning beats gradient boosting](01-when-dl-wins.md)**.
