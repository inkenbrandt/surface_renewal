# srflux/structure.py
"""Structure function utilities used by SR models.

Notes
-----
S_p(Δt) = ⟨ |T(t+Δt) - T(t)|^p ⟩ for even p;
for odd p we keep the signed moment: ⟨ (T(t+Δt) - T(t))^p ⟩.
"""
from __future__ import annotations
import numpy as np
from typing import Iterable, Dict


def _moment(delta: np.ndarray, order: int) -> float:
    if order % 2 == 0:
        return float(np.nanmean(np.abs(delta) ** order))
    else:
        return float(np.nanmean(delta**order))


def structure_functions(
    T: np.ndarray, lags: Iterable[int], orders=(2, 3, 5)
) -> Dict[int, np.ndarray]:
    """Compute structure functions for multiple lags and orders.

    Parameters
    ----------
    T : array-like of float
        Scalar time series (e.g., temperature in K).
    lags : iterable of int
        Positive integer lags in samples.
    orders : iterable of int
        Structure function orders to compute.

    Returns
    -------
    dict[int, np.ndarray]
        Map order -> array of S_order at each lag in the same order.

    Examples
    --------
    The vectorized computation matches the naive per-lag loop:

    >>> import numpy as np
    >>> t = np.arange(200.0)          # linear ramp
    >>> T = np.sin(t / 7.0) + 0.1 * t  # smooth-ish synthetic signal
    >>> lg = [1, 2, 3, 5, 8, 13]
    >>> fast = structure_functions(T, lg, orders=(2, 3, 5))
    >>> ref = {p: np.array(
    ...     [np.nanmean(np.abs(T[k:] - T[:-k]) ** p) if p % 2 == 0
    ...      else np.nanmean((T[k:] - T[:-k]) ** p) for k in lg])
    ...     for p in (2, 3, 5)}
    >>> all(np.allclose(fast[p], ref[p]) for p in (2, 3, 5))
    True
    """
    T = np.asarray(T, float)
    lags = np.asarray(list(lags), int)
    out = {p: np.full(lags.size, np.nan, float) for p in orders}
    n = T.size

    # Only lags with 0 < k < n contribute any samples; the rest stay NaN.
    valid = (lags > 0) & (lags < n)
    if not valid.any():
        return out
    klags = lags[valid]

    # Build every delta array at once as an (nlags x n) matrix instead of
    # looping over lags in Python. delta[i, j] = T[j + k_i] - T[j] where the
    # shifted index is in range, otherwise NaN (the ragged tail per lag).
    # Memory trade-off: this materializes an (nlags x n) float array
    # (e.g. 160 x 36000 ~ 46 MB); acceptable for typical block sizes and
    # far faster than the per-lag Python loop.
    j = np.arange(n)
    idx = j[None, :] + klags[:, None]           # (nlags, n) shifted indices
    in_range = idx < n
    np.clip(idx, 0, n - 1, out=idx)             # avoid out-of-bounds gather
    delta = T[idx] - T[None, :]                 # (nlags, n)
    delta[~in_range] = np.nan

    # Apply the moment in one pass per order over all lags simultaneously,
    # keeping the signed/unsigned convention: even -> |delta|^p, odd -> delta^p.
    for p in orders:
        vals = np.abs(delta) ** p if p % 2 == 0 else delta ** p
        out[p][valid] = np.nanmean(vals, axis=1)
    return out


def estimate_CT2(
    T: np.ndarray,
    *,
    hz: float,
    U: float,
    lag_range_s: tuple[float, float] = (0.05, 0.5),
) -> tuple[float, float]:
    """Estimate the temperature structure parameter ``C_T^2``.

    In the inertial subrange the second-order structure function follows

    .. math:: D_T(r) = C_T^2 \\, r^{2/3},

    where the spatial separation ``r`` is obtained from time lags via Taylor's
    frozen-turbulence hypothesis, ``r = U * dt`` with ``U`` the mean horizontal
    wind speed. The 2/3 slope is theoretical (Kolmogorov scaling), so we fit only
    the intercept ``log(C_T^2)`` with the slope held fixed.

    Parameters
    ----------
    T : array-like of float
        Scalar temperature time series in K.
    hz : float
        Sampling frequency in Hz.
    U : float
        Mean horizontal wind speed in m s⁻¹, used to convert lags to spatial
        separations via Taylor's hypothesis. Values ``<= 0.1`` m s⁻¹ are treated
        as near-calm (Taylor's hypothesis invalid) and yield ``(nan, nan)``.
    lag_range_s : tuple of float, default (0.05, 0.5)
        Inclusive time-lag window (seconds) over which to sample the structure
        function. The default assumes 10–20 Hz data with the inertial subrange
        roughly between ~0.05 and ~0.5 s; users at very rough sites (large
        roughness length, short inertial subrange) may need to shorten it.

    Returns
    -------
    tuple[float, float]
        ``(CT2, r2)`` where ``CT2`` has units K² m⁻²ᐟ³ and ``r2`` is the
        goodness of fit of ``log(S2)`` against the fixed-slope prediction.
        Returns ``(nan, nan)`` if fewer than 4 distinct lags are available,
        if ``U <= 0.1`` m s⁻¹, or if fewer than two finite/positive ``S2``
        values survive.

    References
    ----------
    Wyngaard, J. C., Izumi, Y., & Collins, S. A. (1971). Behavior of the
    refractive-index-structure parameter near the ground. *Journal of the
    Optical Society of America*, 61(12), 1646–1650.
    """
    # (c) Guard near-calm winds: Taylor's frozen-turbulence hypothesis breaks
    # down and r = U*dt is meaningless.
    if not np.isfinite(U) or U <= 0.1:
        return (np.nan, np.nan)

    # (a) Build integer lags spanning lag_range_s at the sampling rate.
    lo_s, hi_s = lag_range_s
    k_lo = max(1, int(np.ceil(lo_s * hz)))
    k_hi = int(np.floor(hi_s * hz))
    if k_hi < k_lo:
        return (np.nan, np.nan)
    lags = np.arange(k_lo, k_hi + 1, dtype=int)
    if lags.size < 4:
        return (np.nan, np.nan)

    # (b) Structure function at those lags (order 2 only).
    S2 = structure_functions(T, lags, orders=(2,))[2]

    # (c) Convert lags to spatial separations via Taylor's hypothesis.
    r = U * (lags / hz)

    # (e) Keep only finite, positive S2 (log undefined otherwise) and positive r.
    good = np.isfinite(S2) & (S2 > 0) & np.isfinite(r) & (r > 0)
    if good.sum() < 2:
        return (np.nan, np.nan)

    log_S2 = np.log(S2[good])
    log_r = np.log(r[good])

    # (d) Fixed 2/3 slope: log(S2) = log(CT2) + (2/3) log(r).
    log_CT2 = float(np.mean(log_S2 - (2.0 / 3.0) * log_r))
    CT2 = float(np.exp(log_CT2))

    # r2 comparing observed log(S2) to the fixed-slope prediction.
    pred = log_CT2 + (2.0 / 3.0) * log_r
    ss_res = float(np.sum((log_S2 - pred) ** 2))
    ss_tot = float(np.sum((log_S2 - np.mean(log_S2)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return (CT2, float(r2))


def pick_optimal_lag(S3: np.ndarray, lags: np.ndarray) -> int:
    """Return the lag (in samples) maximizing |S3| / lag.

    This emphasizes sharp ramps and follows common SR practice.
    """
    S3 = np.asarray(S3, float)
    lags = np.asarray(lags, int)
    score = np.abs(S3) / np.where(lags > 0, lags, np.nan)
    j = int(np.nanargmax(score))
    return int(lags[j])
