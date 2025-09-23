# src/surface_renewal/preprocess/stability.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal

import numpy as np

# We keep the same rotation choices used elsewhere in your codebase
RotationMode = Literal["none", "double", "planar_fit"]

# Reuse your existing implementations for wind rotation & u* from Chen97
# (prevents duplication and ensures identical behavior across the package)
from ..methods.chen97 import estimate_friction_velocity, rotate_wind  # type: ignore
from ..structure import structure_functions, pick_optimal_lag  # type: ignore


@dataclass
class BlockDiagnostics:
    """Container for block-level diagnostics used in SR stability screening.

    Parameters
    ----------
    u_star : float
        Friction velocity (m s⁻¹) computed from rotated covariances of u', v', w'.
    S3_tau : float
        Third-order temperature structure function evaluated at the optimal lag τ*.
    tau_opt : float
        Optimal lag τ* (s), typically maximizing |S3(τ)|/τ or an equivalent criterion.
    stdT : float
        Standard deviation of temperature within the block (K).
    passed : bool, optional
        Whether this block passes the stability screen (filled by `stability_ok`).
    meta : dict, optional
        Extra details for debugging/QA (e.g., chosen lag in samples, indices, etc.).

    Notes
    -----
    The diagnostics follow standard practice for surface-renewal/EC preprocessing:
    1) Despike high-frequency signals,
    2) Rotate winds (planar-fit or double),
    3) Compute u*, ramp signal metrics (S3 at τ*), and temperature variance,
    4) Apply a conservative screen before SR flux estimation.

    This container is intentionally small and serializable for easy logging.
    """
    u_star: float
    S3_tau: float
    tau_opt: float
    stdT: float
    passed: Optional[bool] = None
    meta: Dict[str, Any] = field(default_factory=dict)


def compute_block_diagnostics(
    T: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    *,
    hz: float,
    rotation: RotationMode = "planar_fit",
    lag_window_s: tuple[float, float] = (0.2, 8.0),
) -> BlockDiagnostics:
    """Compute block-level diagnostics for SR stability screening.

    Parameters
    ----------
    T, u, v, w : array_like
        High-frequency time series for temperature and wind components.
        Arrays should be 1-D and aligned (same length).
    hz : float
        Sampling frequency in Hz (e.g., 10 or 20 Hz).
    rotation : {"none", "double", "planar_fit"}, default "planar_fit"
        Rotation mode used for friction velocity computation.
        - "planar_fit": robust, site/period frame (preferred)
        - "double": classic yaw+pitch per block
        - "none": no rotation (discouraged unless inputs are pre-rotated)
    lag_window_s : (float, float), default (0.2, 8.0)
        Inclusive range of lag times (seconds) to scan for τ*.

    Returns
    -------
    BlockDiagnostics
        Container with u*, S3(τ*), τ*, stdT and metadata.

    Notes
    -----
    - τ* selection uses a simple, reproducible rule via `pick_optimal_lag`.
    - `u*` is computed from rotated covariances: u* = [(u'w')² + (v'w')²]^(1/4).
    - Keep lag bounds wide enough to capture dominant ramp scales but not so
      wide that you include very low-frequency trends.
    """
    T = np.asarray(T, float)
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    w = np.asarray(w, float)

    n = T.size
    if n < 4:
        return BlockDiagnostics(u_star=np.nan, S3_tau=np.nan, tau_opt=np.nan, stdT=np.nan,
                                meta={"reason": "too_short"})

    # 1) Lag scan for S3 and τ*
    kmin = max(1, int(lag_window_s[0] * hz))
    kmax = min(n // 4, int(lag_window_s[1] * hz))
    if kmax <= kmin:
        return BlockDiagnostics(u_star=np.nan, S3_tau=np.nan, tau_opt=np.nan, stdT=float(np.nan),
                                meta={"reason": "lag_window_empty",
                                      "kmin": kmin, "kmax": kmax})

    lags = np.arange(kmin, kmax, dtype=int)
    S = structure_functions(T, lags, orders=[3])  # returns dict {3: S3(lag)}
    k_opt = pick_optimal_lag(S[3], lags)          # implementable rule (e.g., argmax |S3|/lag)
    j = int(np.where(lags == k_opt)[0][0])
    S3_tau = float(S[3][j])
    tau_opt = float(k_opt / hz)
    stdT = float(np.nanstd(T))

    # 2) Friction velocity (rotation handled inside estimate_friction_velocity)
    u_star = estimate_friction_velocity(u, v, w, rotation=rotation)

    return BlockDiagnostics(
        u_star=u_star,
        S3_tau=S3_tau,
        tau_opt=tau_opt,
        stdT=stdT,
        meta={"k_opt": int(k_opt), "lags_examined": (int(kmin), int(kmax), int(len(lags)))},
    )


def stability_ok(
    diag: BlockDiagnostics | Dict[str, Any],
    *,
    min_ustar: float = 0.05,
    min_rel_S3: float = 1e-3,
    min_stdT: float = 0.02,
    daytime_only: bool = False,
    Rn_block: Optional[float] = None,
) -> bool:
    """Return True if the block passes SR/EC stability and signal screens.

    Criteria (tunable)
    ------------------
    • u* ≥ `min_ustar`
    • |S3(τ*)| / std(T)^3 ≥ `min_rel_S3`
    • std(T) ≥ `min_stdT`
    • If `daytime_only` and `Rn_block` is provided: Rn_block > 0

    Parameters
    ----------
    diag : BlockDiagnostics or dict
        Diagnostics returned by `compute_block_diagnostics`. A dict with keys
        ``u_star``, ``S3_tau``, ``tau_opt``, and ``stdT`` also works.
    min_ustar : float, default 0.05
        Minimum friction velocity (m s⁻¹).
    min_rel_S3 : float, default 1e-3
        Minimum normalized ramp signal strength |S3(τ*)| / std(T)^3 (dimensionless).
    min_stdT : float, default 0.02
        Minimum temperature standard deviation (K).
    daytime_only : bool, default False
        If True and `Rn_block` is provided, require positive net radiation.
    Rn_block : float, optional
        Mean net radiation (W m⁻²) for this block; used only if `daytime_only=True`.

    Returns
    -------
    bool
        True if all criteria are satisfied.

    Notes
    -----
    Choose thresholds conservatively and validate against your site/season.
    The relative S3 criterion protects against weak or noisy ramp signals.
    """
    # Accept both dataclass and dict input
    if isinstance(diag, BlockDiagnostics):
        u_star = diag.u_star
        S3_tau = diag.S3_tau
        stdT = diag.stdT
    else:
        u_star = float(diag.get("u_star", np.nan))
        S3_tau = float(diag.get("S3_tau", np.nan))
        stdT = float(diag.get("stdT", np.nan))

    # u* and variance checks
    if not np.isfinite(u_star) or u_star < min_ustar:
        return False
    if not np.isfinite(stdT) or stdT < min_stdT:
        return False

    # Relative S3 (dimensionless)
    if not np.isfinite(S3_tau) or stdT <= 0:
        return False
    rel_S3 = abs(S3_tau) / (stdT ** 3)
    if rel_S3 < min_rel_S3:
        return False

    # Optional daytime filter
    if daytime_only and (Rn_block is not None) and not (Rn_block > 0):
        return False

    return True
