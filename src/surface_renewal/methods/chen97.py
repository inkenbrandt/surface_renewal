# src/surface_renewal/methods/chen97.py
from __future__ import annotations

from typing import NamedTuple, Literal, Tuple
import numpy as np

RotationMode = Literal["none", "double", "planar_fit"]


class ChenResult(NamedTuple):
    """Results for Chen (1997)-style SR sensible heat estimation.

    Attributes
    ----------
    tau_opt : float
        Optimal lag τ* in seconds (selected from a bounded scan).
    S3_tau : float
        Third-order temperature structure function evaluated at τ*.
    H : float
        Uncalibrated sensible heat flux (W m⁻²). Apply block-scale alpha
        calibration downstream when matching to a reference (e.g., EC H).
    """
    tau_opt: float
    S3_tau: float
    H: float


# --------------------------------------------------------------------------- #
# Wind rotation helpers (double rotation and small-angle planar-fit)
# --------------------------------------------------------------------------- #

def _rotate_double(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classic two-step (yaw + pitch) rotation to align mean flow with x and ⟨w⟩=0.

    Notes
    -----
    1) Yaw: align mean horizontal wind with the x-axis.
    2) Pitch: remove mean vertical velocity (⟨w⟩ → 0).
    """
    um, vm, wm = np.nanmean(u), np.nanmean(v), np.nanmean(w)
    # Yaw about z
    psi = np.arctan2(vm, um)
    cpsi, spsi = np.cos(-psi), np.sin(-psi)
    u1 = cpsi * u - spsi * v
    v1 = spsi * u + cpsi * v
    w1 = w
    # Pitch about y
    u1m, w1m = np.nanmean(u1), np.nanmean(w1)
    theta = np.arctan2(w1m, u1m)
    cth, sth = np.cos(-theta), np.sin(-theta)
    u2 = cth * u1 + sth * w1
    v2 = v1
    w2 = -sth * u1 + cth * w1
    return u2, v2, w2  # consistent with your current implementation. :contentReference[oaicite:3]{index=3}


def _rotate_planar_fit(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Small-angle planar-fit rotation via Wilczak-style regression.

    Fit the block-mean plane ``w ≈ a0 + a1*u + a2*v`` (least squares), then apply
    the small-angle tilt removal:
        u' = u + a1*w0
        v' = v + a2*w0
        w' = w0 - a1*u - a2*v
    where w0 = w - a0 (remove offset before applying the tilt correction).

    Returns
    -------
    u_p, v_p, w_p : ndarray
        Rotated components.
    """
    U = np.column_stack([np.ones_like(u), u, v])
    coef, *_ = np.linalg.lstsq(U, w, rcond=None)
    a0, a1, a2 = coef
    w0 = w - a0
    u_p = u + a1 * w0
    v_p = v + a2 * w0
    w_p = w0 - a1 * u - a2 * v
    return u_p, v_p, w_p  # matches your prior code path. :contentReference[oaicite:4]{index=4}


def rotate_wind(u: np.ndarray, v: np.ndarray, w: np.ndarray, mode: RotationMode = "planar_fit") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate winds according to the selected mode.

    Parameters
    ----------
    u, v, w : array_like
        Raw wind components (m s⁻¹), aligned and 1-D.
    mode : {"none", "double", "planar_fit"}, default "planar_fit"
        Rotation scheme:
        - "none": no rotation (only if pre-rotated up-front),
        - "double": yaw+pitch per block,
        - "planar_fit": small-angle tilt regression per block.

    Returns
    -------
    u_r, v_r, w_r : ndarray
        Rotated wind components.
    """
    if mode == "none":
        return u, v, w
    if mode == "double":
        return _rotate_double(u, v, w)
    return _rotate_planar_fit(u, v, w)  # default. :contentReference[oaicite:5]{index=5}


# --------------------------------------------------------------------------- #
# Friction velocity
# --------------------------------------------------------------------------- #

def estimate_friction_velocity(u: np.ndarray, v: np.ndarray, w: np.ndarray, rotation: RotationMode = "planar_fit") -> float:
    """Estimate friction velocity u* from rotated covariances.

    u* = [ (u′w′)² + (v′w′)² ]^(1/4)

    Parameters
    ----------
    u, v, w : array_like
        Raw wind components (m s⁻¹).
    rotation : {"none", "double", "planar_fit"}, default "planar_fit"
        Rotation applied prior to covariance to remove tilt and align flow.

    Returns
    -------
    float
        Friction velocity u* (m s⁻¹). Non-negative, NaN-safe.

    Notes
    -----
    Signals are de-meaned within the block before covariance is computed.
    """
    ur, vr, wr = rotate_wind(np.asarray(u, float), np.asarray(v, float), np.asarray(w, float), mode=rotation)
    ur = ur - np.nanmean(ur)
    vr = vr - np.nanmean(vr)
    wr = wr - np.nanmean(wr)
    uw = np.nanmean(ur * wr)
    vw = np.nanmean(vr * wr)
    ustar = (uw ** 2 + vw ** 2) ** 0.25
    return float(max(ustar, 0.0))  # identical definition as your current file. :contentReference[oaicite:6]{index=6}


# --------------------------------------------------------------------------- #
# Chen (1997) sensible heat (uncalibrated) from S3(τ*) and u*
# --------------------------------------------------------------------------- #

def estimate_H_chen(
    T: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    *,
    hz: float,
    rho: float = 1.2,
    cp: float = 1005.0,
    beta: float = 1.0,
    rotation: RotationMode = "planar_fit",
) -> ChenResult:
    """Estimate uncalibrated sensible heat flux using a Chen97-style SR scaling.

    Parameters
    ----------
    T : array_like
        High-frequency temperature series (K or °C after normalization).
    u, v, w : array_like
        Wind components (m s⁻¹), aligned with T.
    hz : float
        Sampling frequency (Hz), e.g., 10 or 20 Hz.
    rho : float, default 1.2
        Air density (kg m⁻³). (You can compute this per-block via ideal gas law.)
    cp : float, default 1005.0
        Specific heat of air at constant pressure (J kg⁻¹ K⁻¹).
    beta : float, default 1.0
        Dimensionless coefficient (absorbed later by block-scale calibration).
    rotation : {"none", "double", "planar_fit"}, default "planar_fit"
        Rotation used for u* computation.

    Returns
    -------
    ChenResult
        Named tuple with (tau_opt[s], S3_tau, H[W m⁻²]).

    Notes
    -----
    - Lags τ are scanned over a bounded window (default 0.2–8 s) to locate τ*
      using a reproducible rule (e.g., ``argmax |S3(τ)| / τ``).
    - The uncalibrated scaling uses u* and S3(τ*) with a dimensional form; in
      practice, apply an **alpha** calibration factor at the block level to
      match an EC reference for H.
    """
    T = np.asarray(T, float)

    # Scan S3 over 0.2–8 s (guarded for short segments)
    kmin = max(1, int(0.2 * hz))
    kmax = min(T.size // 4, int(8.0 * hz))
    lags = np.arange(kmin, max(kmin + 1, kmax), dtype=int)

    # Import here to avoid circular dependencies during package import
    from ..structure import structure_functions, pick_optimal_lag

    S = structure_functions(T, lags, orders=[3])       # {3: S3(lag)}
    k_opt = pick_optimal_lag(S[3], lags)
    tau = float(k_opt) / float(hz)
    # index of k_opt within lags
    S3_tau = float(S[3][int(np.where(lags == k_opt)[0][0])])  # matches your approach. :contentReference[oaicite:7]{index=7}

    # u* from rotated covariances
    ustar = estimate_friction_velocity(u, v, w, rotation=rotation)

    # Dimensional Chen-like scaling (left to be refined by alpha calibration)
    # Guard divisions; keep sign of S3
    tau_term = (tau ** (2.0 / 3.0)) if tau > 0 else np.nan
    H_uncal = rho * cp * beta * ustar * np.cbrt(abs(S3_tau)) * np.sign(S3_tau) / (tau_term + 1e-12)

    return ChenResult(tau_opt=tau, S3_tau=S3_tau, H=float(H_uncal))  # unchanged semantics. :contentReference[oaicite:8]{index=8}
