"""General SR analysis utilities (merges analysis_strfnc.py)."""
from __future__ import annotations

import warnings

import numpy as np

from .wavelet import detect_ramps_wavelet


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
