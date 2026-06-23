# src/surface_renewal/methods/chen97.py
from __future__ import annotations

from typing import NamedTuple, Literal, Optional, Tuple
import numpy as np

RotationMode = Literal["none", "double", "planar_fit"]


class ChenResult(NamedTuple):
    """Results for Chen (1997b)-style SR sensible heat estimation.

    Attributes
    ----------
    tau_opt : float
        Optimal lag ``r_m`` in seconds (selected from a bounded scan via Chen's
        ``r_m`` criterion).
    S3_tau : float
        Third-order temperature structure function evaluated at ``r_m``.
    H : float
        Sensible heat flux (W m⁻²) from the Chen et al. (1997b) formula
        (Mengistu & Savage 2010 Eq. 12). Positive for unstable (S3 < 0),
        negative for stable (S3 > 0). A block-scale calibration may still be
        applied downstream when matching to a reference (e.g., EC H).
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
# Geometric (sublayer) scaling factor G
# --------------------------------------------------------------------------- #

def _geometric_scaling(
    z: float,
    d: float,
    h: Optional[float],
    z_star: Optional[float],
) -> Tuple[float, str]:
    """Return the geometric scaling factor G and the sublayer it assumes.

    Chen et al. (1997b) / Mengistu & Savage (2010) Eq. 12 scale the
    structure-function flux by a height-dependent geometric factor that differs
    between the inertial and roughness sublayers:

    * Inertial sublayer (measurement well above the canopy, ``z - d > z*``)::

          G = z / (z - d) ** (2/3)

    * Roughness sublayer (measurement within ``z*`` of the canopy)::

          G = z / h ** (2/3)

    Parameters
    ----------
    z : float
        Measurement height (m).
    d : float
        Zero-plane displacement height (m).
    h : float or None
        Canopy height (m). Required only in the roughness-sublayer branch.
    z_star : float or None
        Roughness-sublayer top (m). If None it defaults to ``h`` (when given) or
        ``0.0``, so an open/bare surface defaults to the inertial-sublayer form.

    Returns
    -------
    (G, sublayer) : (float, str)
        The geometric factor and which sublayer (``"inertial"`` /
        ``"roughness"``) it assumes.
    """
    zeta = z - d
    if z_star is None:
        z_star = h if h is not None else 0.0

    if zeta > z_star:
        if not (zeta > 0.0):
            raise ValueError(
                f"inertial-sublayer scaling requires z - d > 0; got z={z}, d={d}"
            )
        return z / (zeta ** (2.0 / 3.0)), "inertial"

    if h is None or not (h > 0.0):
        raise ValueError(
            "roughness-sublayer scaling requires a positive canopy height h; "
            f"got h={h} for z={z}, d={d}, z_star={z_star}"
        )
    return z / (h ** (2.0 / 3.0)), "roughness"


# --------------------------------------------------------------------------- #
# Chen (1997b) sensible heat from S3(r_m), u*, and the geometric factor
# --------------------------------------------------------------------------- #

def estimate_H_chen(
    T: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    *,
    hz: float,
    z: float,
    d: float = 0.0,
    h: Optional[float] = None,
    z_star: Optional[float] = None,
    rho: float = 1.2,
    cp: float = 1005.0,
    a_comb: float = 0.4,
    rotation: RotationMode = "planar_fit",
) -> ChenResult:
    """Estimate sensible heat flux using the Chen et al. (1997b) SR formula.

    Implements Mengistu & Savage (2010) Eq. 12::

        H = -a_comb * rho * cp * (S3(r_m) / r_m) ** (1/3) * u_star ** (2/3) * G

    where ``r_m`` (s) is the optimal lag selected by Chen's ``r_m`` criterion
    (:func:`~surface_renewal.structure.pick_optimal_lag`, which maximises
    ``-(S3/r)**(1/3)``), ``S3(r_m)`` is the third-order temperature structure
    function at that lag, ``u_star`` is the friction velocity (note the **2/3**
    exponent), and ``G`` is the height-dependent geometric scaling
    (:func:`_geometric_scaling`).

    Sign convention
    ---------------
    ``np.cbrt`` is a *signed* cube root, so ``(S3 / r_m) ** (1/3)`` keeps the
    sign of ``S3``. Together with the leading minus sign this makes ``H`` come
    out **positive for unstable** stratification (``S3 < 0``, daytime up-ramps)
    and **negative for stable** stratification (``S3 > 0``).

    Parameters
    ----------
    T : array_like
        High-frequency temperature series (K or °C after normalization).
    u, v, w : array_like
        Wind components (m s⁻¹), aligned with T.
    hz : float
        Sampling frequency (Hz), e.g., 10 or 20 Hz.
    z : float
        Measurement height (m).
    d : float, default 0.0
        Zero-plane displacement height (m).
    h : float, optional
        Canopy height (m). Required only when the measurement falls in the
        roughness sublayer (``z - d <= z_star``).
    z_star : float, optional
        Roughness-sublayer top (m). If None, defaults to ``h`` (if given) else
        ``0.0`` so that an open/bare surface uses the inertial-sublayer scaling.
    rho : float, default 1.2
        Air density (kg m⁻³). (You can compute this per-block via ideal gas law.)
    cp : float, default 1005.0
        Specific heat of air at constant pressure (J kg⁻¹ K⁻¹).
    a_comb : float, default 0.4
        Combined coefficient ``alpha * beta**(2/3) * gamma`` (~0.4 per Chen et
        al. 1997b).
    rotation : {"none", "double", "planar_fit"}, default "planar_fit"
        Rotation used for the u* computation.

    Returns
    -------
    ChenResult
        Named tuple ``(tau_opt[s] = r_m, S3_tau = S3(r_m), H[W m⁻²])``.

    Notes
    -----
    Lags are scanned over a bounded window (default 0.2–8 s) to locate ``r_m``.
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
    r_m = float(k_opt) / float(hz)                      # optimal lag in seconds
    # index of k_opt within lags
    S3_rm = float(S[3][int(np.where(lags == k_opt)[0][0])])

    # u* from rotated covariances
    ustar = estimate_friction_velocity(u, v, w, rotation=rotation)

    # Geometric (sublayer) scaling factor G
    G, _sublayer = _geometric_scaling(z, d, h, z_star)

    # Chen (1997b) / Mengistu & Savage (2010) Eq. 12.
    # The signed cube root preserves the sign of S3; the leading minus then
    # yields H > 0 for unstable (S3 < 0) and H < 0 for stable (S3 > 0).
    if not (r_m > 0.0):
        return ChenResult(tau_opt=r_m, S3_tau=S3_rm, H=np.nan)
    H = -a_comb * rho * cp * np.cbrt(S3_rm / r_m) * (ustar ** (2.0 / 3.0)) * G

    return ChenResult(tau_opt=r_m, S3_tau=S3_rm, H=float(H))
