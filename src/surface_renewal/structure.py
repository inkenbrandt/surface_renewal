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
    """
    T = np.asarray(T, float)
    lags = np.asarray(list(lags), int)
    out = {p: np.full(lags.size, np.nan, float) for p in orders}
    n = T.size
    for i, k in enumerate(lags):
        if k <= 0 or k >= n:
            continue
        d = T[k:] - T[:-k]
        for p in orders:
            out[p][i] = _moment(d, p)
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
