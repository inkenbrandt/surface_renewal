"""General SR analysis utilities (merges analysis_strfnc.py)."""
from __future__ import annotations

import numpy as np


def detect_ramps(
    T: np.ndarray,
    fs: int,
    *,
    lags_s: tuple[float, float] = (0.2, 8.0),
) -> dict:
    """Characterize coherent ramp structures in a temperature trace.

    This is a minimal, structure-function based ramp characterization built on
    the same primitives used by the SR flux methods
    (:func:`surface_renewal.structure.structure_functions` and
    :func:`~surface_renewal.structure.pick_optimal_lag`). It recovers the
    characteristic ramp amplitude ``A`` (K) and mean ramp period ``tau`` (s)
    for the block via the Van Atta (1977) / Snyder et al. (1996) linearized
    cubic, then estimates how many such ramps the record spans.

    Parameters
    ----------
    T : np.ndarray
        1-D high-frequency temperature series (K or degC). May contain NaNs.
    fs : int
        Sampling frequency (Hz).
    lags_s : (float, float), optional
        Inclusive range of lag times (s) scanned for the optimal lag.

    Returns
    -------
    dict
        ``{"amp": [...], "tau": [...], "dt_opt": float, "count": int}`` where
        ``amp`` holds the recovered ramp amplitude (K, signed: negative for an
        unstable up-ramp) and ``tau`` the ramp period (s). The lists are empty
        and ``count`` is 0 when no ramp can be resolved (too few samples, a
        degenerate structure function, or a non-positive period).

    Notes
    -----
    The amplitude is the real root of largest magnitude of the depressed cubic
    ``A**3 + p A + q = 0`` with ``p = 10*S2 - S5/S3`` and ``q = 10*S3``
    (Mengistu & Savage 2010, Eqs. 6-9), and ``tau = -A**3 * dt / S3``. The same
    selection rule and equations back :func:`estimate_H_snyder`.
    """
    # Lazy imports mirror snyder.py/chen97.py (cheap module import, no cycles).
    from ..structure import structure_functions, pick_optimal_lag
    from .snyder import _cardano_real_roots

    T = np.asarray(T, dtype=float)
    n = T.size
    empty: dict = {"amp": [], "tau": [], "dt_opt": np.nan, "count": 0}
    if n < max(8, int(fs * 2)):
        return empty

    # Lag grid in samples, bounded by the requested seconds window.
    kmin = max(1, int(lags_s[0] * fs))
    kmax = min(n // 4, int(lags_s[1] * fs))
    if kmax <= kmin:
        return empty
    lags = np.arange(kmin, kmax, dtype=int)

    S = structure_functions(T, lags, orders=[2, 3, 5])
    try:
        k_opt = pick_optimal_lag(S[3], lags)
    except ValueError:
        return empty
    j = int(np.where(lags == k_opt)[0][0])
    S2, S3, S5 = float(S[2][j]), float(S[3][j]), float(S[5][j])
    dt_opt = float(k_opt) / float(fs)

    if not np.isfinite(S3) or S3 == 0.0:
        return {**empty, "dt_opt": dt_opt}

    # Van Atta linearized cubic; select the largest-magnitude real root so the
    # amplitude keeps the correct sign for stable (S3 > 0) blocks.
    p = (10.0 * S2) - (S5 / S3)
    q = 10.0 * S3
    roots = _cardano_real_roots(p, q)
    roots = roots[np.isfinite(roots)]
    if roots.size == 0:
        return {**empty, "dt_opt": dt_opt}
    A = float(roots[np.argmax(np.abs(roots))])

    tau = -(A ** 3) * dt_opt / S3
    if not (np.isfinite(tau) and tau > 0.0):
        return {**empty, "dt_opt": dt_opt}

    # Number of whole ramps the record spans (record length / ramp period).
    count = int((n / float(fs)) // tau)

    return {"amp": [A], "tau": [tau], "dt_opt": dt_opt, "count": count}
