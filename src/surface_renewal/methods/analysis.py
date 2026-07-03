"""General SR analysis utilities (merges analysis_strfnc.py)."""
from __future__ import annotations

import dataclasses
import itertools
import logging
import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from .wavelet import detect_ramps_wavelet

if TYPE_CHECKING:  # avoid an import cycle at module load; pipeline imports methods.*
    from ..pipeline import PipelineConfig

logger = logging.getLogger(__name__)

# Methods that need a measurement height ``cfg.z_m``; silently skipped (with a
# logged warning) by :func:`compare_methods` when it is not configured.
_METHODS_NEEDING_ZM: frozenset[str] = frozenset({"fvs", "castellvi"})

# Default suite: every SR formulation the pipeline knows about.
DEFAULT_COMPARE_METHODS: tuple[str, ...] = (
    "snyder", "chen97", "fvs", "castellvi", "wavelet",
)


def detect_ramps(T: np.ndarray, fs: int) -> dict:
    """Detect ramp events and return stats (amplitude, duration, counts).

    .. deprecated::
        This amplitude-threshold conditional-sampling stub has been superseded by
        the wavelet-based ramp detector
        :func:`surface_renewal.methods.wavelet.detect_ramps_wavelet`
        (Collineau & Brunet 1993), which resolves the dominant ramp scale and
        counts renewals from wavelet zero-crossings. ``detect_ramps`` now
        delegates to it and remains only for backward compatibility; new code
        should call :func:`detect_ramps_wavelet` directly to obtain the full
        :class:`~surface_renewal.methods.wavelet.WaveletRampResult` (signed
        amplitude, dominant period, event count, peak scale and uncalibrated H).

    Parameters
    ----------
    T : np.ndarray
        Temperature time series in Kelvin.
    fs : int
        Sampling frequency in Hz.

    Returns
    -------
    dict
        Dictionary with keys ``"amp"`` (list[float], amplitudes in K),
        ``"tau"`` (list[float], durations in s) and ``"count"`` (int, number of
        detected ramps), reconstructed from the wavelet result for compatibility.
    """
    warnings.warn(
        "detect_ramps is deprecated; use "
        "surface_renewal.methods.wavelet.detect_ramps_wavelet instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    empty = {"amp": [], "tau": [], "count": 0}
    if T is None or fs is None or fs <= 0:
        return empty

    res = detect_ramps_wavelet(np.asarray(T, dtype=float).ravel(), hz=float(fs))
    if not np.isfinite(res.A) or res.n_ramps == 0:
        return {"amp": [], "tau": [], "count": int(res.n_ramps)}

    # Legacy shape: one representative (amplitude, duration) per detected ramp.
    return {
        "amp": [abs(float(res.A))] * res.n_ramps,
        "tau": [float(res.tau)] * res.n_ramps,
        "count": int(res.n_ramps),
    }


def compare_methods(
    data: "pd.DataFrame | str",
    *,
    cfg: "PipelineConfig",
    time_col: Optional[str] = None,
    methods: tuple[str, ...] = DEFAULT_COMPARE_METHODS,
) -> pd.DataFrame:
    """Run several SR methods on the *same* data and return a wide comparison.

    The high-frequency input is read, despiked and rotated exactly once, then
    every requested method is evaluated on that identical preprocessed series
    (via :func:`surface_renewal.pipeline.run_surface_renewal`'s ``preprocessed``
    hook). This makes the block-by-block comparison of the different SR
    formulations fair — the only thing that varies is the flux estimator — and
    avoids repeating the expensive preprocessing once per method.

    Parameters
    ----------
    data : pd.DataFrame or str
        High-frequency table (or path to CSV/Parquet) with at least
        ``["T","u","v","w"]`` (optionally ``["Rn","G"]``).
    cfg : PipelineConfig
        Pipeline configuration. Its ``method`` field is overridden per method;
        everything else (fs, block, rotation, despiking, screening, ``z_m``, …)
        is shared across all methods.
    time_col : str, optional
        Name of the timestamp column if ``data`` lacks a ``DatetimeIndex``.
    methods : tuple of str, default ``("snyder","chen97","fvs","castellvi","wavelet")``
        Methods to evaluate. Any height-dependent method (``fvs``, ``castellvi``)
        is silently skipped, with a logged warning, when ``cfg.z_m`` is ``None``.

    Returns
    -------
    pd.DataFrame
        Block-indexed wide frame with one flux column per successfully-run
        method (``H_snyder``, ``H_chen97``, …) alongside the shared per-block
        diagnostics ``["ustar","stdT","zeta","CT2","passed"]``. ``zeta`` is
        method-dependent (only ``fvs``/``castellvi`` fill it), so it is combined
        across methods, taking the first finite value per block.
    """
    # Local imports keep the module import-cycle-free: pipeline imports methods.*
    from ..pipeline import _ensure_df, _preprocess_df, run_surface_renewal

    # Preprocess once (read → despike → rotate) and reuse for every method.
    df = _ensure_df(data, fs=cfg.fs, time_col=time_col)
    df_prep = _preprocess_df(df, cfg)

    results: dict[str, pd.DataFrame] = {}
    for m in methods:
        if m in _METHODS_NEEDING_ZM and cfg.z_m is None:
            logger.warning(
                "Skipping method %r in compare_methods: it requires cfg.z_m "
                "(measurement height) but z_m is None.", m,
            )
            continue
        cfg_m = dataclasses.replace(cfg, method=m)
        results[m] = run_surface_renewal(cfg=cfg_m, preprocessed=df_prep)

    if not results:
        return pd.DataFrame(
            columns=["ustar", "stdT", "zeta", "CT2", "passed"]
        )

    # Every method ran on the same blocks, so the indexes align. Build the wide
    # frame off the union of block indexes to be safe.
    index = None
    for res in results.values():
        index = res.index if index is None else index.union(res.index)

    wide = pd.DataFrame(index=index)
    for m, res in results.items():
        wide[f"H_{m}"] = res["H_uncal"].reindex(index)

    # Shared diagnostics: ustar/stdT/CT2/passed are method-independent (computed
    # from the identical blocks), so take them from the first result.
    first = next(iter(results.values())).reindex(index)
    wide["ustar"] = first["ustar"]
    wide["stdT"] = first["stdT"]
    wide["CT2"] = first["CT2"]
    # `passed` is a stability screen on the block, identical across methods.
    wide["passed"] = first["passed"].astype(bool)

    # zeta is only filled by height-dependent methods; take the first finite
    # value per block across all methods that reported one.
    zeta = pd.Series(np.nan, index=index)
    for res in results.values():
        if "zeta" in res.columns:
            zeta = zeta.combine_first(res["zeta"].reindex(index))
    wide["zeta"] = zeta

    return wide


def method_agreement(wide: pd.DataFrame) -> pd.DataFrame:
    """Pairwise agreement statistics between the SR methods in ``wide``.

    For every unordered pair of methods (the ``H_*`` columns produced by
    :func:`compare_methods`), the statistics are computed over the blocks where
    both methods are finite *and* the block passed the stability screen
    (``passed == True``):

    - ``slope`` : least-squares slope through the origin of ``H_b`` on ``H_a``
      (``sum(H_a*H_b) / sum(H_a**2)``); ``NaN`` if ``sum(H_a**2) == 0``.
    - ``rmse``  : root-mean-square difference ``sqrt(mean((H_b - H_a)**2))``.
    - ``bias``  : mean signed difference ``mean(H_b - H_a)``.
    - ``N``     : number of blocks contributing to the pair.

    Parameters
    ----------
    wide : pd.DataFrame
        Output of :func:`compare_methods` (needs the ``H_*`` columns and a
        boolean ``passed`` column).

    Returns
    -------
    pd.DataFrame
        Tidy frame indexed by ``(method_a, method_b)`` with columns
        ``["slope","rmse","bias","N"]``. One row per unordered method pair.
    """
    h_cols = [c for c in wide.columns if c.startswith("H_")]
    method_names = [c[len("H_"):] for c in h_cols]

    if "passed" in wide.columns:
        passed = wide["passed"].fillna(False).astype(bool)
    else:
        passed = pd.Series(True, index=wide.index)

    rows: list[dict] = []
    for a, b in itertools.combinations(method_names, 2):
        x = wide[f"H_{a}"]
        y = wide[f"H_{b}"]
        both = passed & np.isfinite(x) & np.isfinite(y)
        xv = x[both].to_numpy(float)
        yv = y[both].to_numpy(float)
        n = int(xv.size)

        denom = float(np.sum(xv * xv))
        slope = float(np.sum(xv * yv) / denom) if denom > 0.0 else np.nan
        if n > 0:
            diff = yv - xv
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            bias = float(np.mean(diff))
        else:
            rmse = np.nan
            bias = np.nan

        rows.append(
            {"method_a": a, "method_b": b,
             "slope": slope, "rmse": rmse, "bias": bias, "N": n}
        )

    out = pd.DataFrame(rows, columns=["method_a", "method_b", "slope", "rmse", "bias", "N"])
    return out.set_index(["method_a", "method_b"])
