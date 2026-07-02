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


def pick_optimal_lag(S3: np.ndarray, lags: np.ndarray) -> int:
    """Return the lag (in samples) maximizing |S3| / lag.

    This emphasizes sharp ramps and follows common SR practice.
    """
    S3 = np.asarray(S3, float)
    lags = np.asarray(lags, int)
    score = np.abs(S3) / np.where(lags > 0, lags, np.nan)
    j = int(np.nanargmax(score))
    return int(lags[j])
