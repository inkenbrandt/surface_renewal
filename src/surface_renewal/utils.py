# src/surface_renewal/utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence, Tuple, Optional

import numpy as np


__all__ = [
    # constants
    "KAPPA", "VON_KARMAN", "GRAV", "R_D", "CP_AIR",
    "constants",
    # robust stats & safe ops
    "mad", "nanmean", "nancov", "demean",
    # signal helpers
    "running_mean", "detrend_linear", "windowed_view", "rolling_std",
    # lag/time helpers
    "seconds_to_samples", "samples_to_seconds", "validate_lags",
    # misc
    "finite_mask", "chunked",
]


# --------------------------------------------------------------------------- #
# Physical constants
# --------------------------------------------------------------------------- #

KAPPA: float = 0.4            # von Kármán constant
VON_KARMAN: float = KAPPA      # alias
GRAV: float = 9.80665          # m s^-2
R_D: float = 287.05            # J kg^-1 K^-1 (dry air gas constant)
CP_AIR: float = 1004.0         # J kg^-1 K^-1 (typical micromet constant)


@dataclass(frozen=True)
class _Constants:
    kappa: float = KAPPA
    g: float = GRAV
    R_d: float = R_D
    c_p: float = CP_AIR


constants = _Constants()


# --------------------------------------------------------------------------- #
# Robust stats & safe ops
# --------------------------------------------------------------------------- #

def mad(x: np.ndarray, *, axis: Optional[int] = None, scale_to_sigma: bool = True) -> np.ndarray:
    """Median absolute deviation (MAD) with optional Gaussian scaling.

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int, optional
        Axis along which to compute the MAD. If None, uses the flattened array.
    scale_to_sigma : bool, default True
        If True, multiplies MAD by 1.4826 to be consistent with σ for Gaussian data.

    Returns
    -------
    np.ndarray
        MAD (or σ-consistent MAD if `scale_to_sigma=True`) along `axis`.
    """
    x = np.asarray(x, float)
    med = np.nanmedian(x, axis=axis, keepdims=True)
    m = np.nanmedian(np.abs(x - med), axis=axis)
    if scale_to_sigma:
        m = 1.4826 * m
    return m


def nanmean(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Shortcut for np.nanmean with float cast."""
    return np.nanmean(np.asarray(x, float), axis=axis)


def nancov(a: np.ndarray, b: np.ndarray) -> float:
    """Nan-safe covariance between two 1-D arrays (demeaned inside).

    Returns
    -------
    float
        Covariance of finite, aligned samples; NaN if <2 valid points.
    """
    aa = np.asarray(a, float)
    bb = np.asarray(b, float)
    m = np.isfinite(aa) & np.isfinite(bb)
    if m.sum() < 2:
        return float("nan")
    aa = aa[m] - aa[m].mean()
    bb = bb[m] - bb[m].mean()
    return float(np.mean(aa * bb))


def demean(x: np.ndarray) -> np.ndarray:
    """Return `x - mean(x)` nan-safely."""
    x = np.asarray(x, float)
    return x - np.nanmean(x)


# --------------------------------------------------------------------------- #
# Signal helpers
# --------------------------------------------------------------------------- #

def running_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Fast running mean using convolution (NaNs ignored via forward-fill trick).

    Parameters
    ----------
    x : array_like
        1-D signal.
    n : int
        Window length (samples). If ≤1, returns a copy of `x`.

    Returns
    -------
    np.ndarray
        Running-mean array (same length as `x`).

    Notes
    -----
    NaNs are linearly interpolated over for the purpose of smoothing only.
    """
    x = np.asarray(x, float).ravel()
    if n <= 1 or x.size == 0:
        return x.copy()

    idx = np.arange(x.size)
    isn = ~np.isfinite(x)
    if isn.any():
        x = np.interp(idx, idx[~isn], x[ ~isn])

    kernel = np.ones(int(n), dtype=float) / float(n)
    y = np.convolve(x, kernel, mode="same")
    return y


def detrend_linear(x: np.ndarray) -> np.ndarray:
    """Remove best-fit line from a 1-D array (NaN-safe).

    Returns
    -------
    np.ndarray
        Detrended series with the same length as `x`.
    """
    x = np.asarray(x, float).ravel()
    n = x.size
    if n < 2:
        return x.copy()
    t = np.arange(n, dtype=float)
    m = np.isfinite(x)
    if m.sum() < 2:
        return np.full_like(x, np.nan)
    A = np.column_stack([t[m], np.ones(m.sum(), float)])
    beta, *_ = np.linalg.lstsq(A, x[m], rcond=None)  # slope, intercept
    fit = beta[0] * t + beta[1]
    out = x.copy()
    out[m] = x[m] - fit[m]
    return out


def windowed_view(x: np.ndarray, window: int) -> np.ndarray:
    """Create an overlapping windowed view (no copies) of a 1-D array.

    Parameters
    ----------
    x : array_like
        1-D signal.
    window : int
        Window length (samples).

    Returns
    -------
    np.ndarray
        Shape (N - window + 1, window) view over `x`.

    Notes
    -----
    Uses stride tricks; modify with care (read-only recommended).
    """
    from numpy.lib.stride_tricks import as_strided

    x = np.asarray(x, float)
    n = x.size
    if window <= 0 or window > n:
        raise ValueError("`window` must be in [1, len(x)].")
    stride = x.strides[0]
    shape = (n - window + 1, window)
    return as_strided(x, shape=shape, strides=(stride, stride))


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation via convolution (O(n)), NaN-safe.

    Parameters
    ----------
    x : array_like
        1-D signal.
    window : int
        Window length (samples). Must be ≥2.

    Returns
    -------
    np.ndarray
        Rolling σ with same length as `x` (edges replicate nearest valid).
    """
    x = np.asarray(x, float).ravel()
    n = x.size
    if window < 2 or n == 0:
        return np.zeros_like(x)

    # Fill for internal stats only
    idx = np.arange(n)
    isn = ~np.isfinite(x)
    if isn.any():
        x = np.interp(idx, idx[~isn], x[~isn])

    k = np.ones(window, dtype=float)
    nwin = np.convolve(np.ones(n), k, "same")
    s = np.convolve(x, k, "same")
    q = np.convolve(x * x, k, "same")
    var = (q - (s * s) / np.maximum(nwin, 1.0)) / np.maximum(nwin - 1.0, 1.0)
    var[var < 0] = 0.0
    return np.sqrt(var, dtype=float)


# --------------------------------------------------------------------------- #
# Lag/time helpers
# --------------------------------------------------------------------------- #

def seconds_to_samples(seconds: float | np.ndarray, fs: float) -> int | np.ndarray:
    """Convert seconds → samples at sampling rate `fs` (rounds to nearest)."""
    s = np.asarray(seconds, float)
    return np.rint(s * float(fs)).astype(int)


def samples_to_seconds(samples: int | np.ndarray, fs: float) -> float | np.ndarray:
    """Convert samples → seconds at sampling rate `fs`."""
    k = np.asarray(samples, float)
    return k / float(fs)


def validate_lags(lags: Sequence[int], n: int) -> np.ndarray:
    """Validate/clip candidate lag samples to lie in [1, n//4] and be unique."""
    lags = np.asarray(lags, int)
    lags = lags[(lags >= 1) & (lags <= max(1, n // 4))]
    return np.unique(lags)


# --------------------------------------------------------------------------- #
# Miscellaneous
# --------------------------------------------------------------------------- #

def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    """Return a boolean mask where all arrays are finite and aligned length."""
    m = None
    for a in arrays:
        a = np.asarray(a, float).ravel()
        m = np.isfinite(a) if m is None else (m & np.isfinite(a))
    return m if m is not None else np.array([], dtype=bool)


def chunked(seq: Sequence, size: int) -> Iterator[Sequence]:
    """Yield fixed-size chunks from a sequence (last chunk may be shorter)."""
    if size <= 0:
        raise ValueError("`size` must be positive.")
    for i in range(0, len(seq), size):
        yield seq[i:i + size]
