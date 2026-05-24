# SciPy — signal, optimize, ndimage

SciPy is the scientific computing layer above NumPy. We've used `scipy.stats` (Module 6) and `scipy.optimize.brentq` (Module 14). This chapter covers the other big subpackages: signal processing, optimisation, and n-dimensional image / array operations.

## `scipy.signal` — signal processing

For filtering, spectral analysis, peak detection on time series:

```python
import numpy as np
from scipy import signal


# Generate test signal: low freq + high freq + noise
t = np.linspace(0, 1, 1000)
sig = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)
noisy = sig + np.random.randn(1000) * 0.2


# Low-pass filter — keep frequencies below 20 Hz
sos = signal.butter(N=4, Wn=20, btype="lowpass", fs=1000, output="sos")
filtered = signal.sosfiltfilt(sos, noisy)
```

`butter(N=4, Wn=20, btype="lowpass", fs=1000)` designs a 4th-order Butterworth filter at 20 Hz cutoff, given 1000 Hz sampling. `sosfiltfilt` applies it forward + backward for zero phase lag.

For high-pass, band-pass, band-stop: same function, different `btype`.

### Spectral analysis

```python
f, Pxx = signal.periodogram(noisy, fs=1000)
# f: frequency bins; Pxx: power at each frequency
```

For non-stationary signals (where the frequency content changes over time):

```python
f, t, Sxx = signal.spectrogram(noisy, fs=1000, nperseg=128)
# Sxx: 2D array of power per time/frequency cell
```

Plot the spectrogram with `plt.pcolormesh(t, f, 10 * np.log10(Sxx))`.

### Peak detection

```python
peaks, props = signal.find_peaks(noisy, height=0.5, distance=10, prominence=0.3)
# Indices where the signal peaks, with constraints
```

Useful for spike detection in tick data, anomaly detection in metrics.

### Convolution and correlation

```python
# Cross-correlation — when does signal A match signal B?
correlation = signal.correlate(a, b, mode="full")
lag = correlation.argmax() - (len(b) - 1)
# Positive lag: a lags b by `lag` samples
```

For "where does this pattern appear in this longer series?" — pattern-matching in time series.

## `scipy.optimize` — function optimisation

For minimising a function:

```python
from scipy.optimize import minimize


def objective(params):
    a, b = params
    return (a - 2) ** 2 + (b + 1) ** 2


result = minimize(objective, x0=[0, 0])
print(result.x)                                  # ~[2, -1]
print(result.fun)                                # ~0
```

The default uses BFGS. For constrained / bounded:

```python
result = minimize(
    objective, x0=[0, 0],
    bounds=[(-5, 5), (-5, 5)],
    constraints=[{"type": "eq", "fun": lambda x: x[0] + x[1] - 1}],   # x0 + x1 = 1
)
```

For global optimisation (avoiding local minima):

```python
from scipy.optimize import differential_evolution


bounds = [(-10, 10), (-10, 10)]
result = differential_evolution(objective, bounds, seed=0)
```

Slower but doesn't get stuck. Used in the course's Heston calibration (Module 15 chapter 3).

### Curve fitting

```python
from scipy.optimize import curve_fit


def model(x, a, b, c):
    return a * np.exp(-b * x) + c


x = np.linspace(0, 5, 50)
y = model(x, 2, 1, 1) + np.random.randn(50) * 0.1

popt, pcov = curve_fit(model, x, y, p0=[1, 1, 1])
print(popt)                                      # estimated [a, b, c]
print(np.sqrt(np.diag(pcov)))                    # 1-sigma uncertainties
```

For "fit this functional form to noisy data with uncertainty estimates." Newton-iteration via Levenberg-Marquardt under the hood.

### Root finding

```python
from scipy.optimize import brentq, fsolve


# Bracketed scalar root
root = brentq(lambda x: x ** 2 - 4, 0, 10)
print(root)                                      # 2.0

# System of equations
def equations(vars):
    x, y = vars
    return [x + y - 3, x * y - 2]

solution = fsolve(equations, [0, 0])
print(solution)                                  # [1, 2] or [2, 1]
```

`brentq` (Module 14) for 1D bracketed roots — robust. `fsolve` (Newton's method) for systems. `bisect` for slow but bulletproof 1D.

## `scipy.ndimage` — multi-dimensional arrays

Image filters and morphology that work on arbitrary-dimensional arrays. For 2D images:

```python
from scipy import ndimage
import numpy as np


img = np.random.rand(100, 100)


# Gaussian smoothing
smooth = ndimage.gaussian_filter(img, sigma=3)


# Median filter (good for salt-and-pepper noise)
denoised = ndimage.median_filter(img, size=3)


# Sobel edge detection
edges_x = ndimage.sobel(img, axis=0)
edges_y = ndimage.sobel(img, axis=1)
edges = np.hypot(edges_x, edges_y)


# Labeled regions (connected components)
binary = img > 0.5
labels, n_labels = ndimage.label(binary)
print(f"{n_labels} connected regions")


# Centroids of labelled regions
centroids = ndimage.center_of_mass(img, labels, range(1, n_labels + 1))
```

Useful for any "image-like" data: actual images, heatmaps, 2D data fields. Works in N-D for volume data (3D MRI scans, etc.).

### Morphological operations

```python
from scipy.ndimage import binary_dilation, binary_erosion, binary_opening, binary_closing


# Grow / shrink
dilated = binary_dilation(binary, iterations=2)
eroded = binary_erosion(binary, iterations=2)


# Open = erode then dilate (removes small specks)
# Close = dilate then erode (fills small holes)
cleaned = binary_opening(binary, iterations=1)
```

For cleaning up binary masks from thresholded images.

## `scipy.interpolate` — interpolation

```python
from scipy.interpolate import interp1d, CubicSpline


x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 4, 9, 16])

f = CubicSpline(x, y)
print(f(2.5))                                    # ~6.25 (between data points)


# Multi-dim
from scipy.interpolate import RegularGridInterpolator

# 2D grid lookup
xs = np.linspace(0, 1, 10)
ys = np.linspace(0, 1, 10)
values = np.random.rand(10, 10)
interp = RegularGridInterpolator((xs, ys), values)
print(interp([0.5, 0.5]))                        # interpolated at (0.5, 0.5)
```

For vol-surface interpolation, look up table queries, etc.

## `scipy.spatial` — geometric structures

```python
from scipy.spatial import KDTree, distance, ConvexHull


# Nearest-neighbour queries
points = np.random.rand(10000, 3)
tree = KDTree(points)
query = np.array([0.5, 0.5, 0.5])
distances, indices = tree.query(query, k=5)         # 5 nearest neighbours


# All-pairs distances (for small sets)
d = distance.cdist(points[:100], points[:100])      # 100x100 distance matrix


# Convex hull
hull = ConvexHull(points)
print(hull.vertices)                                 # indices of hull corners
```

`KDTree` is essential for "find the k nearest things" at scale.

## A worked example: low-pass filter for noisy sensor data

```python
import numpy as np
from scipy import signal


def lowpass_filter(data: np.ndarray, cutoff_hz: float, sample_hz: float, order: int = 4):
    sos = signal.butter(order, cutoff_hz, btype="lowpass", fs=sample_hz, output="sos")
    return signal.sosfiltfilt(sos, data)


sensor_data = ...                                # 1 kHz sampled accelerometer
clean = lowpass_filter(sensor_data, cutoff_hz=20, sample_hz=1000)
```

For any real-world signal: low-pass before downstream processing eliminates the high-frequency noise that biases derived metrics.

## Pitfalls

!!! warning "Filter design without considering phase"
    A forward-only filter (`signal.sosfilt`) introduces phase lag — the filtered output is shifted in time. For offline processing, use `sosfiltfilt` (forward + backward — zero phase). For real-time, accept the lag.

!!! warning "Optimisers that converge to local minima"
    `minimize` with a non-convex objective gets stuck. For global, use `differential_evolution` or `basinhopping`.

!!! warning "Spline interpolation extrapolation"
    Cubic splines extrapolate wildly outside the data range. Always check the requested point is within the data; clip if needed.

!!! warning "Naming collisions with numpy"
    `scipy.signal.convolve` vs `numpy.convolve` — slightly different APIs. Pin the import explicitly.

## Bottom line

For SciPy:

- **`signal`** for filtering, FFT, spectrograms.
- **`optimize`** for minimisation, root-finding, curve-fitting.
- **`ndimage`** for image / array filters.
- **`interpolate`** for interpolation.
- **`spatial`** for nearest-neighbours, distances, hulls.

## End of Module 27 and the course

You now have every module — from Python foundations through trading-specific strategies to general operational engineering and scientific corners of the stdlib. The next page is the **Cookbook**: 100+ short, copy-pasteable recipes for everyday tasks.

Continue to the **[Cookbook](../cookbook/index.md)**.
