# Wavelets and the Hilbert transform

Most financial time series have structure at multiple scales — a tick stream has noise at milliseconds, mean-reversion at minutes, trend at hours, regimes at weeks. Fourier analysis tells you which frequencies are present *on average* over your whole sample. Wavelet analysis tells you which frequencies are present *at each point in time*.

This chapter is a short, applied tour. Don't memorise the math; learn what to reach for and when.

## The big idea

A wavelet is a localised wave — a short, decaying oscillation. By correlating a wavelet with your signal at every position and every scale, you build a **time-frequency representation**: a 2D map of "how much of this frequency is present at this moment".

The two main classes:

- **Continuous wavelet transform (CWT)** — dense scale-time map. Great for visualisation and feature engineering.
- **Discrete wavelet transform (DWT)** — orthogonal decomposition into coarse-and-detail components at dyadic scales. Great for denoising and compression.

## CWT, with `pywt`

```python
import numpy as np
import pywt

# A noisy signal with two regimes: high-freq then low-freq
t = np.linspace(0, 1, 1000)
sig = np.where(t < 0.5,
               np.sin(2 * np.pi * 50 * t),
               np.sin(2 * np.pi * 10 * t))
sig += 0.3 * np.random.randn(len(t))

scales = np.arange(1, 128)
coeffs, freqs = pywt.cwt(sig, scales, wavelet="morl")
# coeffs.shape == (len(scales), len(sig))
# coeffs[i, j] = magnitude of frequency `freqs[i]` at time `j`
```

The `coeffs` matrix is the scalogram. Plot `np.abs(coeffs)` as a heatmap and you can see — visually — when high-frequency activity gives way to low-frequency activity.

For finance, the CWT is a great **feature**: you can take `np.abs(coeffs)` at a few representative scales as inputs to a model, capturing "energy at intraday vs daily vs weekly horizons" without manual lookback selection.

## DWT for denoising

```python
import pywt
import numpy as np

def wavelet_denoise(signal: np.ndarray, wavelet: str = "db4", level: int = 4) -> np.ndarray:
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # Universal threshold (Donoho-Johnstone)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thresh = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, thresh, mode="soft") for c in coeffs[1:]]
    return pywt.waverec(coeffs_thresh, wavelet)[: len(signal)]

clean = wavelet_denoise(noisy_price_series)
```

Wavelet denoising is less aggressive than a moving average — it preserves sharp transitions while smoothing the noise floor. For event-driven signal generation, it can be a better preprocessing step.

!!! danger "Wavelet denoising and look-ahead"
    `wavedec` operates on the *whole* series. For a backtest signal, you must do **causal** wavelet denoising — only use information available up to time $t$. That means rolling the denoise on a moving window, which is much more expensive. For research, offline denoising is fine.

## The Hilbert transform — instantaneous phase and amplitude

For a real signal $x(t)$, the analytic signal is

$$
x_a(t) = x(t) + i \mathcal{H}\{x(t)\}
$$

where $\mathcal{H}$ is the Hilbert transform. From $x_a$ you extract:

- **Instantaneous amplitude**: $|x_a(t)|$ — the envelope.
- **Instantaneous phase**: $\arg x_a(t)$ — the local angle of the oscillation.
- **Instantaneous frequency**: derivative of the phase.

```python
from scipy.signal import hilbert

# Detrended cycle component of a price series
import numpy as np
y = prices - prices.rolling(window=200).mean()
analytic = hilbert(y.dropna().values)
envelope = np.abs(analytic)
phase    = np.unwrap(np.angle(analytic))
inst_freq = np.diff(phase) / (2 * np.pi)        # cycles per bar
```

What's it useful for?

- **Cycle identification.** If `inst_freq` settles around 1/20, your series has a 20-bar dominant cycle.
- **Phase-based entries.** Buy when phase is at the "trough" (multiples of $2\pi$); sell at peaks.
- **Envelope-targeted vol.** The envelope is a smooth, time-varying amplitude that you can target like vol.

The Hilbert transform is **non-causal** as implemented in scipy — it uses the entire series. For online use, there's a streaming version based on FIR filters; for research, use the offline one.

## Empirical Mode Decomposition (EMD)

A purely data-driven decomposition (no fixed basis). The EMD finds **intrinsic mode functions** (IMFs) — narrow-band oscillatory components — by repeatedly extracting and removing the envelope.

```python
# pip install EMD-signal
from PyEMD import EMD

emd = EMD()
imfs = emd(prices.values)             # shape (k_imfs, N)
```

The first few IMFs are the high-frequency noise; the last is the long-term trend; the middle ones are the genuine cycles. Combined with Hilbert (HHT: Hilbert-Huang Transform), you get a time-frequency representation that adapts to non-stationary data.

EMD is **interesting but unstable** — running it twice on slightly different data can produce slightly different decompositions. Treat it as an exploratory tool, not a production signal source.

## Multifractal analysis

Real financial returns aren't a single fractal — they're multifractal. The local Hurst exponent (chapter 1) varies in time. Tools like the **multifractal detrended fluctuation analysis (MF-DFA)** quantify this:

```python
# Quick sketch — the MFDFA package implements this properly
# pip install MFDFA
from MFDFA import MFDFA

q = np.array([-5, -3, -1, 0, 1, 3, 5])         # range of moments
scales = np.logspace(1, 3.5, 30).astype(int)
F = MFDFA(returns.values, scales, q=q)
# F has shape (len(scales), len(q)); slopes give the generalised Hurst exponents
```

For a monofractal (Gaussian random walk), the slope is constant in $q$. For multifractal data (real returns), it varies — and the *width* of the multifractal spectrum is a measure of intermittency.

This is genuinely **frontier-track** material. Modules 11 and 17 use multifractal width as a regime-fragility feature.

## When wavelets are the right tool — and when not

**Right:**
- Visualising time-varying frequency content.
- Multi-scale feature extraction for ML.
- Denoising signals where you care about edges.

**Not right:**
- Simple smoothing. Use rolling mean / EWM.
- Anything where you need a one-line answer. Wavelets are a toolbox, not a single technique.
- Real-time signal generation with causal constraints (without serious engineering).

## Pitfalls

!!! warning "Wavelet choice matters less than you think"
    `db4`, `sym4`, `coif3` — for most denoising tasks they're interchangeable. Pick one, move on.

!!! warning "Edge effects"
    Wavelet transforms can't see past the start/end. The first few and last few coefficients are noisy. Use `pywt.Modes.symmetric` or pad the signal.

!!! warning "Hilbert transform on noisy data is meaningless"
    The instantaneous phase of pure noise is random. Always pre-filter (band-pass) before computing Hilbert phase.

## Bottom line

Wavelets and Hilbert are specialised tools. Use them when:

- You suspect a cyclic component you want to extract.
- You want time-varying frequency features for an ML model.
- You're denoising a signal where edges matter.

Otherwise stick to the simpler tools (EWM, rolling, FFT) — they're easier to reason about and almost always good enough.

Continue to **[Change-point detection](06-change-points.md)**.
