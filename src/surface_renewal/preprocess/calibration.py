# src/surface_renewal/preprocess/calibration.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Iterable, Literal

import numpy as np
import pandas as pd

__all__ = [
    "to_kelvin",
    "to_celsius",
    "normalize_temperature",
    "apply_linear_calibration",
    "Calibration",
    "fit_gain_offset",
    "align_on_index",
    "rho_air_ideal",
    "cp_air_const",
]


# --------------------------------------------------------------------------- #
# Unit helpers
# --------------------------------------------------------------------------- #

def to_kelvin(T: pd.Series | np.ndarray) -> pd.Series:
    """Convert temperature from °C to K if values appear to be in °C.

    Parameters
    ----------
    T : array-like
        Temperature series.

    Returns
    -------
    pd.Series
        Temperature in Kelvin.

    Notes
    -----
    Heuristic: if median(T) < 200 → treat as °C and add 273.15.
    """
    s = pd.Series(T, dtype=float)
    if s.median(skipna=True) < 200:
        return s + 273.15
    return s


def to_celsius(T: pd.Series | np.ndarray) -> pd.Series:
    """Convert temperature from K to °C if values appear to be in K."""
    s = pd.Series(T, dtype=float)
    if s.median(skipna=True) > 200:
        return s - 273.15
    return s


def normalize_temperature(df: pd.DataFrame, col: str = "T", to: Literal["K", "C"] = "K") -> pd.DataFrame:
    """Ensure a dataframe temperature column is in a requested unit.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with column `col`.
    col : str, default "T"
        Temperature column name.
    to : {"K","C"}, default "K"
        Target unit.

    Returns
    -------
    pd.DataFrame
        Copy with normalized temperature.
    """
    out = df.copy()
    if to.upper() == "K":
        out[col] = to_kelvin(out[col])
    else:
        out[col] = to_celsius(out[col])
    return out


# --------------------------------------------------------------------------- #
# Generic linear calibration for sensor channels
# --------------------------------------------------------------------------- #

def apply_linear_calibration(
    series: pd.Series | np.ndarray,
    *,
    slope: float = 1.0,
    intercept: float = 0.0,
) -> pd.Series:
    """Apply y = slope * x + intercept to a 1-D series."""
    x = pd.Series(series, dtype=float)
    return slope * x + intercept


def fit_gain_offset(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    fit_intercept: bool = True,
    robust: bool = False,
) -> Tuple[float, float]:
    """Estimate `slope` and `intercept` for y ≈ slope * x + intercept.

    Parameters
    ----------
    x, y : array-like
        Predictors (raw readings) and targets (truth/reference).
    fit_intercept : bool, default True
        If False, intercept is constrained to 0 (pure gain/alpha).
    robust : bool, default False
        If True, use median-based slope on inliers (simple Huber-like trimming).

    Returns
    -------
    slope, intercept : float, float
        Fitted calibration parameters (intercept=0 if fit_intercept=False).
    """
    xx = pd.Series(x, dtype=float)
    yy = pd.Series(y, dtype=float)
    m = xx.notna() & yy.notna()
    xx = xx[m].to_numpy()
    yy = yy[m].to_numpy()
    if xx.size < 2:
        return np.nan, np.nan

    if not robust:
        if fit_intercept:
            # Ordinary least squares with intercept
            X = np.column_stack([xx, np.ones_like(xx)])
            beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
            slope, intercept = float(beta[0]), float(beta[1])
        else:
            # Constrained through origin
            slope = float(np.dot(xx, yy) / np.dot(xx, xx))
            intercept = 0.0
        return slope, intercept

    # Robust (trimmed) version: drop extreme residuals iteratively
    slope, intercept = (1.0, 0.0) if fit_intercept else (1.0, 0.0)
    for _ in range(3):
        if fit_intercept:
            X = np.column_stack([xx, np.ones_like(xx)])
            beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
            slope, intercept = float(beta[0]), float(beta[1])
            resid = yy - (slope * xx + intercept)
        else:
            slope = float(np.dot(xx, yy) / np.dot(xx, xx))
            intercept = 0.0
            resid = yy - slope * xx
        sig = np.nanstd(resid)
        if not np.isfinite(sig) or sig == 0:
            break
        keep = np.abs(resid) < 2.5 * sig
        if keep.sum() < max(10, int(0.5 * len(xx))):
            break
        xx, yy = xx[keep], yy[keep]
    return slope, intercept


def align_on_index(
    left: pd.Series | pd.DataFrame,
    right: pd.Series | pd.DataFrame,
    *,
    how: Literal["inner", "left", "right"] = "inner",
) -> Tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame]:
    """Align two time series/dataframes on their indices using pandas join logic."""
    L = left.copy()
    R = right.copy()
    if not isinstance(L.index, pd.DatetimeIndex) or not isinstance(R.index, pd.DatetimeIndex):
        # fall back to intersection of raw indices
        ix = L.index.intersection(R.index)
        return L.loc[ix], R.loc[ix]
    if how == "inner":
        ix = L.index.intersection(R.index)
        return L.loc[ix], R.loc[ix]
    if how == "left":
        return L, R.reindex(L.index)
    if how == "right":
        return L.reindex(R.index), R
    raise ValueError("how must be 'inner', 'left', or 'right'.")


# --------------------------------------------------------------------------- #
# SR-style flux calibration (alpha) and air properties
# --------------------------------------------------------------------------- #

@dataclass
class Calibration:
    """Multiplicative (and optional additive) calibration for model outputs.

    Parameters
    ----------
    alpha : float, default 1.0
        Multiplicative scale (e.g., apply to modeled H to match EC reference).
    beta : float, default 0.0
        Optional additive offset (rarely used; prefer alpha-only when possible).
    name : str, optional
        Free-form label for provenance (e.g., "SR↔EC 2023-05").

    Notes
    -----
    In surface renewal workflows, it’s common to fit a single **alpha** to map
    uncalibrated SR **H** to a reference sensible heat (e.g., EC H) at the
    **block** resolution, keeping beta=0 to avoid biasing low-flux periods.
    """
    alpha: float = 1.0
    beta: float = 0.0
    name: Optional[str] = None

    def apply(self, series: pd.Series | np.ndarray) -> pd.Series:
        """Return calibrated series: alpha * x + beta."""
        return pd.Series(series, dtype=float) * self.alpha + self.beta

    @classmethod
    def from_reference(
        cls,
        model_series: pd.Series,
        reference_series: pd.Series,
        *,
        fit_intercept: bool = False,
        robust: bool = True,
        name: Optional[str] = None,
    ) -> "Calibration":
        """Fit calibration to map `model_series` → `reference_series`.

        Parameters
        ----------
        model_series : pd.Series
            Modeled output (e.g., uncalibrated SR H) indexed by block end time.
        reference_series : pd.Series
            Reference observations at the same block resolution (e.g., EC H).
        fit_intercept : bool, default False
            If False, constrain beta=0 (pure multiplicative alpha).
        robust : bool, default True
            Use trimmed fit to limit influence of outliers.
        name : str, optional
            Label saved in the object.

        Returns
        -------
        Calibration
            Instance with fitted alpha (and beta, if allowed).
        """
        m, r = align_on_index(model_series.dropna(), reference_series.dropna(), how="inner")
        if len(m) < 5:
            return cls(alpha=np.nan, beta=np.nan, name=name)

        slope, intercept = fit_gain_offset(m, r, fit_intercept=fit_intercept, robust=robust)
        alpha = float(slope) if np.isfinite(slope) else np.nan
        beta = float(intercept) if (fit_intercept and np.isfinite(intercept)) else 0.0
        return cls(alpha=alpha, beta=beta, name=name)


def rho_air_ideal(
    T_K: pd.Series | np.ndarray,
    P_Pa: pd.Series | np.ndarray = 101325.0,
    q_kgkg: pd.Series | np.ndarray | None = None,
) -> pd.Series:
    """Air density via ideal gas law: ρ = P / (R_d T_v).

    Parameters
    ----------
    T_K : array-like
        Air temperature in Kelvin.
    P_Pa : array-like or float, default 101325
        Air pressure in Pa.
    q_kgkg : array-like, optional
        Specific humidity (kg kg⁻¹). If provided, density is computed for moist
        air using the virtual temperature T_v = T_K * (1 + 0.608 * q). If None,
        falls back to dry-air density (R_d = 287.05 J kg⁻¹ K⁻¹).

    Returns
    -------
    pd.Series
        Air density in kg m⁻³.

    Notes
    -----
    Uses R_d = 287.05 J kg⁻¹ K⁻¹ (dry air). For humid air, density is reduced
    via the virtual temperature: at T=300 K and q=0.015 kg/kg, ρ is ~0.9% lower
    than the dry-air value.
    """
    R_d = 287.05
    T = pd.Series(T_K, dtype=float)
    P = pd.Series(P_Pa, dtype=float).reindex_like(T).fillna(101325.0)
    if q_kgkg is None:
        T_v = T
    else:
        q = pd.Series(q_kgkg, dtype=float).reindex_like(T).fillna(0.0)
        T_v = T * (1.0 + 0.608 * q)
    return P / (R_d * T_v)


def cp_air_const(T_K: pd.Series | np.ndarray | float = 300.0) -> pd.Series:
    """Return c_p for air (J kg⁻¹ K⁻¹).

    Notes
    -----
    A constant **1004–1006** J kg⁻¹ K⁻¹ is typical for micrometeorology. Use a
    constant unless you have strong reason to model T-dependence.
    """
    T = pd.Series(T_K, dtype=float)  # placeholder in case you add T dependence later
    return pd.Series(np.full_like(T, 1004.0, dtype=float), index=T.index)
