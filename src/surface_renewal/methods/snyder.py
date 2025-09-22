# src/surface_renewal/methods/snyder.py
from __future__ import annotations

from typing import NamedTuple, Tuple
import numpy as np


class SnyderResult(NamedTuple):
    """Results for Snyder-style cubic-ramp sensible heat estimation.

    Attributes
    ----------
    A : float
        Ramp amplitude (K) recovered from the cubic relation.
    tau : float
        Mean time between ramps τ (s).
    H : float
        Uncalibrated sensible heat flux (W m⁻²). Apply block-scale calibration later.
    dt_opt : float
        Optimal lag Δt* (s) used to evaluate S2, S3, S5.
    """
    A: float
    tau: float
    H: float
    dt_opt: float


# --------------------------------------------------------------------------- #
# Cardano solver (depressed cubic x^3 + p x + q = 0)
# --------------------------------------------------------------------------- #

def _cardano_real_roots(p: float, q: float) -> np.ndarray:
    """Return all real roots of x^3 + p x + q = 0 (sorted).

    Parameters
    ----------
    p, q : float
        Coefficients of the depressed cubic.

    Returns
    -------
    np.ndarray
        Array of size 1 or 3 containing the real roots (ascending).

    Notes
    -----
    We evaluate the discriminant Δ = (q/2)^2 + (p/3)^3.  If Δ > 0, one real root;
    if Δ == 0, multiple real roots (at least a double root); if Δ < 0, three real roots.
    """
    p = float(p)
    q = float(q)

    # Discriminant in classic form
    Delta = (q * 0.5) ** 2 + (p / 3.0) ** 3

    if Delta > 0.0:
        # One real root
        sqrtD = np.sqrt(Delta)
        u = np.cbrt(-q * 0.5 + sqrtD)
        v = np.cbrt(-q * 0.5 - sqrtD)
        return np.array([u + v], dtype=float)

    if Delta == 0.0:
        # Multiple real roots (at least two equal)
        u = np.cbrt(-q * 0.5)
        # Roots: 2u, -u, -u
        return np.sort(np.array([2.0 * u, -u, -u], dtype=float))

    # Three distinct real roots: trigonometric form
    # Write p = -3 r^2, q = -2 r^3 cos(3θ),  r = 2 sqrt(-p/3) / 2
    r = 2.0 * np.sqrt(-p / 3.0)
    phi = np.arccos(np.clip((-q / 2.0) / ((-p / 3.0) ** 1.5), -1.0, 1.0)) / 3.0
    x1 = r * np.cos(phi)
    x2 = r * np.cos(phi + 2.0 * np.pi / 3.0)
    x3 = r * np.cos(phi + 4.0 * np.pi / 3.0)
    return np.sort(np.array([x1, x2, x3], dtype=float))


# --------------------------------------------------------------------------- #
# Snyder 1996 cubic-ramp workflow (Van Atta linearized model)
# --------------------------------------------------------------------------- #

def estimate_H_snyder(
    T: np.ndarray,
    *,
    hz: float,
    rho: float = 1.2,
    cp: float = 1005.0,
    lags_s: tuple[float, float] = (0.2, 8.0),
) -> SnyderResult:
    """Estimate uncalibrated sensible heat using Snyder’s cubic-ramp recovery.

    Steps (block level)
    -------------------
    1) Compute S2, S3, S5 of temperature across a lag scan (Δt ∈ [0.2, 8] s by default).
    2) Choose Δt* by a reproducible rule (e.g., argmax |S3(Δt)| / Δt).
    3) Form the depressed cubic for **A** (ramp amplitude):
           A^3 + p A + q = 0,
       with
           p = 10·S2  −  (S5 / S3),    q = 10·S3    (evaluated at Δt*).
    4) Select **A** as the **maximum real root**.
    5) Recover τ from the model relation:
           τ = − (A^3) · Δt* / S3(Δt*).
    6) Compute H = ρ c_p (A / τ).  (Apply block-scale calibration downstream.)

    Parameters
    ----------
    T : array_like
        High-frequency temperature series (K or °C after normalization).
    hz : float
        Sampling frequency (Hz), e.g., 10 or 20 Hz.
    rho : float, default 1.2
        Air density (kg m⁻³).
    cp : float, default 1005.0
        Specific heat at constant pressure (J kg⁻¹ K⁻¹).
    lags_s : (float, float), default (0.2, 8.0)
        Inclusive range for lag times (seconds) to scan.

    Returns
    -------
    SnyderResult
        (A, τ, H, Δt*).  Values are NaN if the recovery is degenerate.

    Notes
    -----
    - The implementation follows your current draft (S2/S3/S5, cubic coefficients,
      maximum real root choice, τ relation, H scaling). Keep an external, site/period
      calibration (alpha) to reconcile H with EC where desired.
    """
    T = np.asarray(T, float)
    n = T.size
    if n < max(64, int(hz * 2)):
        return SnyderResult(A=np.nan, tau=np.nan, H=np.nan, dt_opt=np.nan)

    # Build lag grid in samples
    kmin = max(1, int(lags_s[0] * hz))
    kmax = min(n // 4, int(lags_s[1] * hz))
    if kmax <= kmin:
        return SnyderResult(A=np.nan, tau=np.nan, H=np.nan, dt_opt=np.nan)
    lags = np.arange(kmin, kmax, dtype=int)

    # Structure functions at candidate lags
    # (import here to avoid circulars at package import time)
    from ..structure import structure_functions, pick_optimal_lag
    S = structure_functions(T, lags, orders=[2, 3, 5])

    k_opt = pick_optimal_lag(S[3], lags)  # e.g., argmax |S3|/lag
    j = int(np.where(lags == k_opt)[0][0])
    S2 = float(S[2][j]); S3 = float(S[3][j]); S5 = float(S[5][j])
    dt_opt = float(k_opt) / float(hz)

    # Guard degenerate divisions
    if not np.isfinite(S3) or S3 == 0.0:
        return SnyderResult(A=np.nan, tau=np.nan, H=np.nan, dt_opt=dt_opt)

    # Cubic coefficients (Van Atta linearized relations)
    p = (10.0 * S2) - (S5 / S3)
    q = 10.0 * S3

    roots = _cardano_real_roots(p, q)
    roots = roots[np.isfinite(roots)]
    if roots.size == 0:
        return SnyderResult(A=np.nan, tau=np.nan, H=np.nan, dt_opt=dt_opt)

    # Practice in Snyder/Chen implementations: take the maximum real root
    A = float(np.max(roots))

    # Recover τ; must be positive and finite
    tau = - (A ** 3) * dt_opt / S3
    if not (np.isfinite(tau) and (tau > 0.0)):
        return SnyderResult(A=np.nan, tau=np.nan, H=np.nan, dt_opt=dt_opt)

    # Sensible heat (uncalibrated).  Calibrate later at block scale if desired.
    dTdt = A / tau
    H = rho * cp * dTdt

    return SnyderResult(A=A, tau=float(tau), H=float(H), dt_opt=dt_opt)


# --------------------------------------------------------------------------- #
# Optional utility for direct (S2, S3, S5) → (A, τ) recovery
# --------------------------------------------------------------------------- #

def solve_ramp_from_structure(S2: float, S3: float, S5: float) -> Tuple[float, float]:
    """Recover (A, τ) directly from structure function moments.

    Parameters
    ----------
    S2, S3, S5 : float
        Second, third, and fifth-order temperature structure functions
        evaluated at a chosen lag (e.g., Δt*).

    Returns
    -------
    A, tau : float, float
        Ramp amplitude and period.  Returns (nan, nan) if degenerate.

    Notes
    -----
    This helper simply wraps the Cardano solve and τ relation for already-
    computed (S2, S3, S5).  It mirrors the same equations used in
    `estimate_H_snyder`.
    """
    if not (np.isfinite(S2) and np.isfinite(S3) and np.isfinite(S5)) or S3 == 0.0:
        return np.nan, np.nan

    p = (10.0 * S2) - (S5 / S3)
    q = 10.0 * S3
    roots = _cardano_real_roots(p, q)
    roots = roots[np.isfinite(roots)]
    if roots.size == 0:
        return np.nan, np.nan

    A = float(np.max(roots))
    tau = - (A ** 3) * 1.0 / S3  # here Δt*=1 by construction; scale externally if needed
    return (A, tau if np.isfinite(tau) and (tau > 0.0) else np.nan)
