# src/surface_renewal/preprocess/rotation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Literal

import numpy as np
import pandas as pd

__all__ = [
    "RotationResult",
    "planar_fit",
    "double_rotation",
    "apply_rotation_matrix",
    "friction_velocity",
    "build_planar_fit_matrix",
]


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #

@dataclass
class RotationResult:
    """Container for rotation outputs.

    Attributes
    ----------
    df : pd.DataFrame
        DataFrame copy with added rotated columns: ``u_r, v_r, w_r``.
    R : np.ndarray
        3x3 rotation matrix (rows are new basis vectors in the original frame).
        For a vector ``v`` in the original frame, the rotated vector is:
        ``v_r = R @ v``.
    meta : dict
        Additional metadata (means, normal vector, projection info).
    """
    df: pd.DataFrame
    R: np.ndarray
    meta: dict


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return a unit-norm vector, guarding against near-zero norms."""
    n = float(np.linalg.norm(v))
    if n < eps:
        return v.copy()
    return v / n


def _project_onto_plane(v: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Project vector ``v`` onto plane with unit normal ``k``."""
    return v - np.dot(v, k) * k


def build_planar_fit_matrix(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Build a site/period planar-fit rotation matrix from (u, v, w).

    This implements a standard planar-fit approach:

    1. Fit a plane ``w = a + b*u + c*v`` via least squares.
    2. The plane's unit normal is proportional to ``n = (b, c, -1)``.
    3. Define the new vertical axis ``k'`` as the plane normal (upward).
    4. Define the new streamwise axis ``x'`` as the projection of the mean wind
       onto the plane.
    5. ``y' = k' × x'`` to complete a right-handed basis.
    6. Build rotation matrix ``R`` with rows ``x', y', k'`` so
       ``v_r = R @ v`` gives components in the rotated frame.

    Parameters
    ----------
    u, v, w : np.ndarray
        1-D arrays of wind components (m s⁻¹). NaNs are ignored in the fit.

    Returns
    -------
    R : (3, 3) ndarray
        Rotation matrix to go from original to planar-fit coordinates.
    meta : dict
        Diagnostic info (coeffs, means, normals).

    Notes
    -----
    - If the mean horizontal wind is extremely weak, a fallback horizontal axis
      is chosen to keep the basis stable.
    """
    # Mask NaNs consistently
    m = ~(np.isnan(u) | np.isnan(v) | np.isnan(w))
    u0, v0, w0 = u[m], v[m], w[m]

    # Least-squares plane: w = a + b*u + c*v
    X = np.column_stack([np.ones_like(u0), u0, v0])
    # Solve (X^T X) beta = X^T w  → beta = [a, b, c]
    beta, *_ = np.linalg.lstsq(X, w0, rcond=None)
    a, b, c = beta

    # Plane unit normal (upward): proportional to (b, c, -1)
    n = np.array([b, c, -1.0], dtype=float)
    k_prime = _normalize(n)

    # Mean wind vector
    U_mean = np.array([np.nanmean(u), np.nanmean(v), np.nanmean(w)], dtype=float)

    # Streamwise axis is the projection of mean wind onto the plane
    x_prime = _project_onto_plane(U_mean, k_prime)
    x_prime = _normalize(x_prime)

    # If mean wind is nearly perpendicular to plane (rare), choose fallback
    if not np.all(np.isfinite(x_prime)) or np.linalg.norm(x_prime) < 1e-6:
        # fallback: horizontal axis orthogonal to k' with preference along x
        fallback = np.array([1.0, 0.0, 0.0])
        x_prime = _project_onto_plane(fallback, k_prime)
        x_prime = _normalize(x_prime)

    # Cross to get y'
    y_prime = np.cross(k_prime, x_prime)
    y_prime = _normalize(y_prime)

    # Assemble rotation matrix with rows as the new basis vectors
    R = np.vstack([x_prime, y_prime, k_prime])

    meta = {
        "plane_coeffs": {"a": float(a), "b": float(b), "c": float(c)},
        "U_mean": U_mean,
        "k_prime": k_prime,
        "x_prime": x_prime,
        "y_prime": y_prime,
    }
    return R, meta


def apply_rotation_matrix(
    df: pd.DataFrame,
    R: np.ndarray,
    u_col: str = "u",
    v_col: str = "v",
    w_col: str = "w",
    suffix: str = "_r",
) -> pd.DataFrame:
    """Apply a 3×3 rotation matrix to (u, v, w) and return a copy with rotated columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns for wind components.
    R : (3, 3) ndarray
        Rotation matrix where rows are new basis vectors in original coordinates.
    u_col, v_col, w_col : str, default "u", "v", "w"
        Column names for the raw wind components.
    suffix : str, default "_r"
        Suffix for rotated columns (u_r, v_r, w_r).

    Returns
    -------
    pd.DataFrame
        Copy of `df` with new columns ``u_r, v_r, w_r``.
    """
    out = df.copy()
    U = np.column_stack([out[u_col].to_numpy(float),
                         out[v_col].to_numpy(float),
                         out[w_col].to_numpy(float)])
    # v_r = R @ v (apply to each row)
    Ur = (R @ U.T).T
    out[f"u{suffix}"] = Ur[:, 0]
    out[f"v{suffix}"] = Ur[:, 1]
    out[f"w{suffix}"] = Ur[:, 2]
    return out


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def planar_fit(
    df: pd.DataFrame,
    *,
    u_col: str = "u",
    v_col: str = "v",
    w_col: str = "w",
    fit_over: Optional[pd.Index] = None,
    persist_matrix_in_meta: bool = True,
) -> RotationResult:
    """Perform planar-fit rotation on (u, v, w) and return rotated components.

    Parameters
    ----------
    df : pd.DataFrame
        Input high-frequency data. Despike **before** calling this function.
    u_col, v_col, w_col : str, default "u", "v", "w"
        Column names for wind components (m s⁻¹).
    fit_over : pd.Index, optional
        Index specifying which rows to use for building the rotation matrix.
        If None, uses all rows in `df`. For a site-wide matrix, pass a long,
        well-mixed period; for period-specific fits, pass a mask/index subset.
    persist_matrix_in_meta : bool, default True
        If True, attach the rotation matrix and diagnostics in ``.attrs['planar_fit']``.

    Returns
    -------
    RotationResult
        A container with the rotated DataFrame, the 3×3 matrix, and metadata.

    Notes
    -----
    - Planar-fit gives a **fixed** rotation matrix (over the chosen fit period),
      unlike double-rotation which enforces zero mean ``v_r`` and ``w_r`` per
      averaging block. SR pipelines often prefer planar-fit to maintain a stable
      frame for covariance estimates (u*, w'T', etc.).
    """
    # Rows used to estimate the matrix
    if fit_over is None:
        idx = df.index
    else:
        idx = fit_over

    u = df.loc[idx, u_col].to_numpy(float)
    v = df.loc[idx, v_col].to_numpy(float)
    w = df.loc[idx, w_col].to_numpy(float)

    R, meta = build_planar_fit_matrix(u, v, w)
    out = apply_rotation_matrix(df, R, u_col=u_col, v_col=v_col, w_col=w_col)

    if persist_matrix_in_meta:
        # Attach site/period matrix so downstream code can reuse it
        meta_store = {"R": R, **meta}
        out.attrs["planar_fit"] = meta_store

    return RotationResult(df=out, R=R, meta=meta)


def double_rotation(
    df: pd.DataFrame,
    *,
    u_col: str = "u",
    v_col: str = "v",
    w_col: str = "w",
    by: Optional[Literal["resample", "none"]] = "none",
    period: str = "30min",
    suffix: str = "_r",
) -> pd.DataFrame:
    """Classic two-step (double) rotation to align mean flow and zero mean ``w``.

    Step 1 rotates about ``w`` to align mean wind with the x-axis (v̄' = 0).
    Step 2 rotates about the new y-axis to enforce w̄'' = 0.

    Parameters
    ----------
    df : pd.DataFrame
        Input high-frequency data.
    u_col, v_col, w_col : str, default "u", "v", "w"
        Column names for wind components (m s⁻¹).
    by : {"resample", "none"}, optional
        If "none", compute one rotation for the full dataset.
        If "resample", compute and apply a separate rotation per `period`.
    period : str, default "30min"
        Resample period if `by="resample"`.
    suffix : str, default "_r"
        Suffix for rotated columns.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with rotated columns (u_r, v_r, w_r).

    Notes
    -----
    - Double rotation is sensitive to averaging choices and is less stable over
      heterogeneous terrain. Prefer planar-fit for site-level rotation matrices.
    """
    def _rot_block(block: pd.DataFrame) -> pd.DataFrame:
        u = block[u_col].to_numpy(float)
        v = block[v_col].to_numpy(float)
        w = block[w_col].to_numpy(float)

        # Means
        ubar, vbar, wbar = np.nanmean(u), np.nanmean(v), np.nanmean(w)

        # Step 1: rotate about w so that v̄' = 0
        alpha = np.arctan2(vbar, ubar)  # yaw
        ca, sa = np.cos(alpha), np.sin(alpha)
        Rz = np.array([[ ca,  sa, 0.0],
                       [-sa,  ca, 0.0],
                       [0.0, 0.0, 1.0]])

        UVW1 = (Rz @ np.column_stack([u, v, w]).T).T
        u1, v1, w1 = UVW1[:, 0], UVW1[:, 1], UVW1[:, 2]

        # Step 2: rotate about y so that w̄'' = 0
        w1bar = np.nanmean(w1)
        u1bar = np.nanmean(u1)
        beta = np.arctan2(w1bar, u1bar)  # pitch
        cb, sb = np.cos(beta), np.sin(beta)
        Ry = np.array([[ cb, 0.0, -sb],
                       [0.0, 1.0,  0.0],
                       [ sb, 0.0,  cb]])

        UVW2 = (Ry @ UVW1.T).T
        block = block.copy()
        block[f"u{suffix}"] = UVW2[:, 0]
        block[f"v{suffix}"] = UVW2[:, 1]
        block[f"w{suffix}"] = UVW2[:, 2]
        return block

    if by == "resample":
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Resample-based double rotation requires a DatetimeIndex.")
        parts = []
        for _, grp in df.groupby(pd.Grouper(freq=period)):
            if len(grp) == 0:
                continue
            parts.append(_rot_block(grp))
        return pd.concat(parts).sort_index()
    else:
        return _rot_block(df)


def friction_velocity(
    df: pd.DataFrame,
    *,
    u_col: str = "u_r",
    v_col: str = "v_r",
    w_col: str = "w_r",
    method: Literal["rolling", "resample", "global"] = "global",
    window: Optional[int | str] = None,
    center: bool = True,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Compute friction velocity u* from rotated covariances.

    u* is defined as:
    ``u* = [ (u'w')² + (v'w')² ]^(1/4)`` where primes denote fluctuations.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing **rotated** wind components.
    u_col, v_col, w_col : str, default "u_r", "v_r", "w_r"
        Column names for rotated components.
    method : {"rolling", "resample", "global"}, default "global"
        - "global": one u* over the entire frame.
        - "resample": u* per time block (e.g., "30min") using DatetimeIndex.
        - "rolling": rolling u* with integer/window spec.
    window : int or str, optional
        For "rolling": integer window (samples) or offset string (e.g., "600s").
        For "resample": period string (e.g., "30min").
        Ignored for "global".
    center : bool, default True
        Center the rolling window (if method="rolling" and window is integer).
    min_periods : int, optional
        Minimum samples per window (rolling) to compute a covariance.

    Returns
    -------
    pd.Series
        u* time series aligned to df.index for rolling/resample,
        or a length-1 series for the global case.

    Notes
    -----
    - Ensure **rotation** (planar-fit or double) is applied before calling this.
    - For "resample", we de-mean per block and compute covariances within the block.
    """
    def _cov(a: pd.Series, b: pd.Series) -> float:
        aa = a.to_numpy(float)
        bb = b.to_numpy(float)
        m = ~(np.isnan(aa) | np.isnan(bb))
        if m.sum() < 2:
            return np.nan
        aa = aa[m] - np.nanmean(aa[m])
        bb = bb[m] - np.nanmean(bb[m])
        return float(np.nanmean(aa * bb))

    ur = df[u_col]
    vr = df[v_col]
    wr = df[w_col]

    if method == "global":
        uw = _cov(ur, wr)
        vw = _cov(vr, wr)
        val = (uw**2 + vw**2) ** 0.25
        return pd.Series([val], index=[df.index[0] if len(df) else 0])

    if method == "resample":
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Resample method requires a DatetimeIndex.")
        if window is None:
            window = "30min"
        def _block_u_star(g: pd.DataFrame) -> float:
            uw = _cov(g[u_col], g[w_col])
            vw = _cov(g[v_col], g[w_col])
            return (uw**2 + vw**2) ** 0.25
        return df.resample(window).apply(_block_u_star)

    if method == "rolling":
        if window is None:
            raise ValueError("Provide a window for rolling u*.")
        # Use rolling with custom covariance via apply on a helper DataFrame
        tmp = df[[u_col, v_col, w_col]].copy()
        if isinstance(window, str):
            roll = tmp.rolling(window)
        else:
            roll = tmp.rolling(window=window, center=center, min_periods=min_periods or max(10, int(0.5 * (window if isinstance(window, int) else 10))))
        def _u_star_from_window(win: pd.DataFrame) -> float:
            uw = _cov(win[u_col], win[w_col])
            vw = _cov(win[v_col], win[w_col])
            return (uw**2 + vw**2) ** 0.25
        return roll.apply(_u_star_from_window, raw=False).iloc[:, 0]  # align to index

    raise ValueError("method must be 'rolling', 'resample', or 'global'.")
