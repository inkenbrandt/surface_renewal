# src/surface_renewal/preprocess/despike.py
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Iterable, Literal, Tuple, Dict, Optional

from scipy.signal import windows
from scipy.interpolate import interp1d

__all__ = [
    "HampelResult",
    "hampel",
    "despike_gaussian",
    "despike",  # backward-compat wrapper
    "interpolate_over_nans",
    "despike_dataframe",
    "velocity_temperature_consistency",
]


# ----------------------------- Data containers ------------------------------ #

@dataclass
class HampelResult:
    """Container for Hampel filtering results.

    Attributes
    ----------
    series : pd.Series
        The filtered series (with spikes replaced by NaN by default).
    mask : pd.Series
        Boolean mask where True indicates detected spikes.
    threshold : pd.Series
        Time-varying MAD-based threshold used for detection.
    """
    series: pd.Series
    mask: pd.Series
    threshold: pd.Series


# ----------------------------- Core algorithms ------------------------------ #

def hampel(
    series: pd.Series,
    window: int = 11,
    n_sigmas: float = 3.0,
    center: bool = True,
    return_replaced: bool = False,
    replace_with: Literal["nan", "median"] = "nan",
) -> HampelResult:
    """Hampel filter for spike detection and optional replacement.

    Parameters
    ----------
    series : pd.Series
        Input time series (numeric). Should be 1-D with a monotonic index.
    window : int, default 11
        Odd window length for median / MAD calculation.
    n_sigmas : float, default 3.0
        Points with |x - median| > n_sigmas * (1.4826 * MAD) are flagged.
        (1.4826 scales MAD to be consistent with the standard deviation for Gaussian data.)
    center : bool, default True
        Center the rolling window on each point (recommended).
    return_replaced : bool, default False
        If True and `replace_with="median"`, returns the *replaced* series in result.series.
        Otherwise spikes are replaced by NaN.
    replace_with : {"nan", "median"}, default "nan"
        Replacement strategy for flagged spikes.

    Returns
    -------
    HampelResult
        Container with the filtered series, spike mask, and the threshold.

    Notes
    -----
    - Robust to outliers; recommended as a first-pass detector for high-frequency EC signals.
    - Consistent with QA/QC practice in micrometeorology: detect spikes, propagate NaNs,
      later *optionally* interpolate within short gaps (see `interpolate_over_nans`).

    See Also
    --------
    interpolate_over_nans, despike_gaussian, despike_dataframe
    """
    x = series.astype(float).copy()

    if window % 2 == 0:
        window += 1  # enforce odd

    # Rolling median and MAD (median absolute deviation)
    med = x.rolling(window=window, center=center, min_periods=1).median()
    mad = (x - med).abs().rolling(window=window, center=center, min_periods=1).median()

    # Scale MAD to be std-consistent (Gaussian)
    sigma = 1.4826 * mad
    threshold = n_sigmas * sigma

    # Spike mask
    mask = (x - med).abs() > threshold
    out = x.copy()

    if replace_with == "nan":
        out[mask] = np.nan
    elif replace_with == "median":
        out[mask] = med[mask]
    else:
        raise ValueError('replace_with must be "nan" or "median".')

    return HampelResult(series=out if return_replaced else out, mask=mask, threshold=threshold)


def _moving_window_std(signal: np.ndarray, w: int) -> np.ndarray:
    """Compute moving-window standard deviation using convolution (O(n)).

    Ensures non-negative variance due to numerical error.
    """
    if w < 2:
        # degenerate case
        return np.zeros_like(signal, dtype=float)

    kernel = np.ones(w, dtype=float)
    n = np.convolve(np.ones_like(signal, dtype=float), kernel, mode="same")
    s = np.convolve(signal, kernel, mode="same")
    q = np.convolve(signal * signal, kernel, mode="same")
    var = (q - (s * s) / n) / np.maximum(n - 1.0, 1.0)
    var[var < 0] = 0.0
    return np.sqrt(var, dtype=float)


def despike_gaussian(
    data: np.ndarray,
    nw: int,
    sig: float,
    buffer: int,
    timestamps: Optional[np.ndarray] = None,
    interpolate: Optional[Literal["linear", "nearest", "cubic"]] = None,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Despike a 1D signal using Gaussian convolution and a moving-std threshold.

    Parameters
    ----------
    data : np.ndarray
        1-D numeric array.
    nw : int
        Window length for both the Gaussian smoother and the moving std.
        (Larger -> smoother baseline and broader detection.)
    sig : float
        Number of standard deviations above local baseline to flag as spike.
    buffer : int
        Number of *adjacent* samples to set to NaN around flagged spikes
        (applied symmetrically via convolution with a ones kernel).
    timestamps : np.ndarray, optional
        Optional 1-D array aligned with `data` for interpolation.
    interpolate : {"linear", "nearest", "cubic"}, optional
        If provided, fill NaNs (from despiking) by interpolating over timestamps.

    Returns
    -------
    data_ds : np.ndarray
        Despiked array (spikes + buffer set to NaN; optionally interpolated).
    ns : int
        Number of flagged samples (before buffering).
    index : np.ndarray
        Boolean array where True marks a flagged spike (before buffering).

    Notes
    -----
    - This is a cleaned, vectorized version of your MATLAB-style routine translated to Python.
    - We first form a smoothed baseline via Gaussian convolution, then compare residuals
      against a locally estimated moving standard deviation.
    - End effects are minimized by ignoring partial windows in detection.

    See Also
    --------
    hampel, despike_dataframe, interpolate_over_nans
    """
    x = np.asarray(data, dtype=float).ravel()
    n = x.size
    if n == 0:
        return x.copy(), 0, np.zeros(0, dtype=bool)

    # Preserve original NaNs and fill temporally for internal filtering
    orig_nan = np.isnan(x)
    xs = np.arange(n)
    if orig_nan.any():
        x = np.interp(xs, xs[~orig_nan], x[~orig_nan])

    # Gaussian window (std=(nw-1)/2 imitates MATLAB gausswin default alpha=1)
    if nw < 3:
        nw = 3
    win = windows.gaussian(nw, std=(nw - 1) / 2.0)
    win /= win.sum()

    # Smooth baseline and residuals
    baseline = np.convolve(x, win, mode="same")
    hw = int(np.ceil(len(win) / 2))
    valid = np.ones(n, dtype=bool)
    valid[:hw] = False
    valid[-hw:] = False

    local_std = _moving_window_std(x, nw)
    thresh = sig * local_std
    resid = np.zeros_like(x)
    resid[valid] = x[valid] - baseline[valid]

    # Flag spikes (protect against zero std)
    index = np.zeros(n, dtype=bool)
    nz = thresh > 0
    index[nz] = np.abs(resid[nz]) > thresh[nz]
    ns = int(index.sum())

    # Buffer flagged region
    if buffer > 0:
        kernel = np.ones(2 * buffer + 1, dtype=int)
        buffered = np.convolve(index.astype(int), kernel, mode="same") > 0
    else:
        buffered = index.copy()

    out = x.copy()
    out[buffered] = np.nan

    # Restore original NaNs
    out[orig_nan] = np.nan

    # Optional interpolation over the gaps we just created
    if interpolate is not None:
        out = interpolate_over_nans(out, timestamps=timestamps, kind=interpolate)

    return out, ns, index


# ------------------------------- Public API -------------------------------- #

def despike(
    data: np.ndarray,
    nw: int,
    sig: float,
    buffer: int,
    *args,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Backward-compatible wrapper around :func:`despike_gaussian`.

    This preserves your original call style:
    - `despike(data, nw, sig, buffer, 'interp', timestamps, method)`
    - `despike(data, nw, sig, buffer, 'plot1'|'plot2', timestamps)`

    Plotting is not performed here (plots belong in notebooks / analysis scripts).
    Interpolation is supported.

    Parameters
    ----------
    data, nw, sig, buffer : see :func:`despike_gaussian`.
    *args : tuple
        Legacy-style options:
        - ('interp', timestamps, method) → interpolate over NaNs
        - ('plot1'|'plot2', [timestamps]) → ignored (no-op, kept for compatibility)

    Returns
    -------
    data_ds, ns, index : see :func:`despike_gaussian`.
    """
    interpolate = None
    timestamps = None
    if len(args) >= 1:
        if args[0] == "interp":
            timestamps = args[1] if len(args) > 1 else None
            interpolate = args[2] if len(args) > 2 else "linear"
        # 'plot1'/'plot2' are acknowledged but intentionally ignored in this clean API

    return despike_gaussian(
        data=np.asarray(data),
        nw=int(nw),
        sig=float(sig),
        buffer=int(buffer),
        timestamps=timestamps,
        interpolate=interpolate,
    )


def interpolate_over_nans(
    arr: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    kind: Literal["linear", "nearest", "cubic"] = "linear",
    max_gap: Optional[int] = None,
) -> np.ndarray:
    """Interpolate over NaNs in a 1-D array with optional gap length control.

    Parameters
    ----------
    arr : np.ndarray
        1-D array possibly containing NaNs.
    timestamps : np.ndarray, optional
        1-D array of the same length as `arr` giving x-values for interpolation.
        If None, uses index positions [0..n-1].
    kind : {"linear", "nearest", "cubic"}, default "linear"
        Interpolation scheme passed to `scipy.interpolate.interp1d`.
    max_gap : int, optional
        If provided, only interpolate consecutive NaN runs with length <= max_gap.
        Longer gaps remain NaN.

    Returns
    -------
    np.ndarray
        Interpolated array.

    Notes
    -----
    Use a *small* `max_gap` (e.g., at 10–20 Hz, <= 1–2 seconds) to avoid biasing turbulence
    statistics, consistent with SR preprocessing best practices.
    """
    x = np.asarray(arr, dtype=float).copy()
    n = x.size
    if n == 0:
        return x

    t = np.arange(n) if timestamps is None else np.asarray(timestamps)
    isn = np.isnan(x)
    if not isn.any():
        return x

    # Optionally restrict interpolation to short gaps
    if max_gap is not None and max_gap > 0:
        # identify nan runs
        run_starts = np.where(np.diff(np.r_[False, isn, False]))[0]
        runs = list(zip(run_starts[0::2], run_starts[1::2]))
        # mask only short runs to be filled
        fill_mask = np.zeros(n, dtype=bool)
        for a, b in runs:
            if (b - a) <= max_gap:
                fill_mask[a:b] = True
        ind = fill_mask
    else:
        ind = isn

    if (~isn).sum() < 2:
        # insufficient points to interpolate
        return x

    f = interp1d(t[~isn], x[~isn], kind=kind, bounds_error=False, fill_value="extrapolate")
    x[ind] = f(t[ind])
    return x


def despike_dataframe(
    df: pd.DataFrame,
    columns: Iterable[str],
    method: Literal["hampel", "gaussian"] = "hampel",
    *,
    window: int = 11,
    n_sigmas: float = 3.0,
    nw: int = 201,
    sig: float = 4.0,
    buffer: int = 3,
    timestamps_col: Optional[str] = None,
    interpolate: Optional[Literal["linear", "nearest", "cubic"]] = None,
    max_gap: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """Despike multiple columns of a DataFrame with a consistent policy.

    Parameters
    ----------
    df : pd.DataFrame
        Input frame with a DatetimeIndex or numeric index.
    columns : iterable of str
        Column names to despike (e.g., ["T", "u", "v", "w"]).
    method : {"hampel", "gaussian"}, default "hampel"
        Which detector to use.
    window, n_sigmas : see :func:`hampel` (used if method="hampel").
    nw, sig, buffer : see :func:`despike_gaussian` (used if method="gaussian").
    timestamps_col : str, optional
        Column in `df` to use as timestamps (1-D numeric). If None, uses index position.
    interpolate : {"linear", "nearest", "cubic"}, optional
        Interpolate spikes (NaNs) after detection.
    max_gap : int, optional
        Maximum consecutive NaNs to interpolate (samples).

    Returns
    -------
    df_out : pd.DataFrame
        Copy of `df` with despiked columns.
    masks : dict[str, pd.Series]
        Dictionary mapping column name → boolean mask of flagged spikes.

    Notes
    -----
    The pipeline should **despike first**, then perform rotation (planar fit), u*,
    stability (z/L), and SR calculations—consistent with the SR literature’s
    “preprocess → rotate → screen → flux” workflow.
    """
    df_out = df.copy()
    masks: Dict[str, pd.Series] = {}
    t = df_out.index.values if timestamps_col is None else df_out[timestamps_col].values

    for col in columns:
        x = df_out[col].to_numpy(dtype=float, copy=True)

        if method == "hampel":
            res = hampel(pd.Series(x, index=df_out.index), window=window, n_sigmas=n_sigmas)
            x_out = res.series.to_numpy()
            mask = res.mask
        elif method == "gaussian":
            x_out, _, idx = despike_gaussian(
                data=x, nw=nw, sig=sig, buffer=buffer, timestamps=t, interpolate=None
            )
            mask = pd.Series(idx, index=df_out.index)
        else:
            raise ValueError('method must be "hampel" or "gaussian".')

        if interpolate is not None:
            x_out = interpolate_over_nans(x_out, timestamps=t, kind=interpolate, max_gap=max_gap)

        df_out[col] = x_out
        masks[col] = mask

    return df_out, masks


def velocity_temperature_consistency(
    df: pd.DataFrame,
    t_col: str = "T",
    u_cols: Iterable[str] = ("u", "v", "w"),
    t_abs_limit: Optional[Tuple[float, float]] = (-60.0, 60.0),
    uvw_abs_limit: Optional[float] = 40.0,
) -> pd.Series:
    """Flag unphysical combinations in temperature–velocity records (non-destructive).

    Parameters
    ----------
    df : pd.DataFrame
        Input with columns for temperature and wind components (pre-despiked).
    t_col : str, default "T"
        Temperature column name (°C or K after normalization).
    u_cols : iterable of str, default ("u", "v", "w")
        Wind component column names (m s⁻¹).
    t_abs_limit : (float, float), optional
        Absolute min/max limits for temperature.
    uvw_abs_limit : float, optional
        Absolute limit for |u|, |v|, or |w|.

    Returns
    -------
    pd.Series
        Boolean mask where True indicates *suspicious* (to be excluded downstream).

    Notes
    -----
    This is an integrity *screen* only—no values are overwritten. Use it to build
    QA/QC flags alongside spike masks. Keep the policy conservative and site-aware.
    """
    mask = pd.Series(False, index=df.index)

    if t_col in df and t_abs_limit is not None:
        lo, hi = t_abs_limit
        mask |= (df[t_col] < lo) | (df[t_col] > hi)

    for c in u_cols:
        if c in df and uvw_abs_limit is not None:
            mask |= df[c].abs() > uvw_abs_limit

    # Additional cross-channel heuristics could be added here (e.g., w large while T flat).
    return mask
