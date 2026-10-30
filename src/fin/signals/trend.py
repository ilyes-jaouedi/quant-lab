"""Trend signals as linear filters.

Every classic trend rule is the sign of a linear filter applied to past log returns r
(Levine & Pedersen, 2016, "Which Trend Is Your Friend?"). With x the log price:

- momentum over n days:   x_t - x_{t-n}           = sum_{k<n} r_{t-k}            (box kernel)
- price vs moving average: x_t - sum_j h_j x_{t-j} = sum_k (sum_{j>k} h_j) r_{t-k}

so a smoother h on prices (SMA, EMA) defines a kernel on returns, its tail sum. Comparing trend
rules is comparing kernels: their lag (centre of mass), noise gain and frequency response.

Kernels here are causal FIR weights indexed by lag: ``w[k]`` multiplies the value k days ago.
"""

import numpy as np
from scipy import signal


def sma_kernel(n: int) -> np.ndarray:
    """Simple moving average of the last n values."""
    return np.full(n, 1.0 / n)


def ema_kernel(span: float, length: int | None = None) -> np.ndarray:
    """Exponential moving average, alpha = 2 / (span + 1), truncated and renormalized.

    With this convention the EMA of span n has the same lag as the SMA of length n: (n - 1) / 2.
    """
    alpha = 2.0 / (span + 1.0)
    length = length or int(np.ceil(10 * span))
    w = alpha * (1 - alpha) ** np.arange(length)
    return w / w.sum()


def momentum_kernel(n: int) -> np.ndarray:
    """Kernel on returns of the n-day log return x_t - x_{t-n}."""
    return np.ones(n)


def price_minus_smoother(h: np.ndarray) -> np.ndarray:
    """Kernel on returns of x_t - (h * x)_t, for a smoother h on prices summing to 1."""
    h = np.asarray(h, dtype=float) / np.sum(h)
    tail = h[::-1].cumsum()[::-1]  # tail[k] = sum_{j>=k} h_j
    return tail[1:]  # w[k] = sum_{j>k} h_j


def crossover_kernel(fast: np.ndarray, slow: np.ndarray) -> np.ndarray:
    """Kernel on returns of (fast * x) - (slow * x), e.g. a moving-average crossover."""
    n = max(len(fast), len(slow))
    f = np.pad(price_minus_smoother(fast), (0, n - len(fast)))
    s = np.pad(price_minus_smoother(slow), (0, n - len(slow)))
    return s - f


def apply_kernel(r: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Causal filtering: out[t] = sum_k w[k] * r[t-k]; NaN until the kernel is filled."""
    r = np.asarray(r, dtype=float)
    out = np.convolve(r, w, mode="full")[: len(r)]
    out[: len(w) - 1] = np.nan
    return out


def lag(w: np.ndarray) -> float:
    """Centre of mass of a kernel, in samples: its average delay."""
    w = np.asarray(w, dtype=float)
    return float(np.sum(np.arange(len(w)) * w) / np.sum(w))


def noise_gain(w: np.ndarray) -> float:
    """Output variance per unit input variance for white noise, with the kernel scaled to unit sum.

    Lower means smoother. Comparing kernels at equal lag isolates the shape.
    """
    w = np.asarray(w, dtype=float) / np.sum(w)
    return float(np.sum(w**2))


def frequency_response(w: np.ndarray, n_freq: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """(frequency in cycles per day, complex response) of a kernel, normalized to 1 at DC."""
    w = np.asarray(w, dtype=float) / np.sum(w)
    freq, h = signal.freqz(w, worN=n_freq, fs=1.0)
    return freq, h


def group_delay(w: np.ndarray, n_freq: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """(frequency in cycles per day, group delay in days) of a kernel."""
    freq, gd = signal.group_delay((np.asarray(w, dtype=float), [1.0]), w=n_freq, fs=1.0)
    return freq, gd
