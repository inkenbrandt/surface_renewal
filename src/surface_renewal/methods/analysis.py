"""General SR analysis utilities (merges analysis_strfnc.py)."""
from __future__ import annotations
import numpy as np
import pandas as pd

def detect_ramps(T: np.ndarray, fs: int) -> dict:
    """Detect ramp events and return stats (amplitude, duration, counts).

    Conditional-sampling ramp detection
    -----------------------------------
    Surface-renewal theory models the scalar (here temperature) time series as a
    sequence of coherent "ramp" structures: a gradual enrichment/depletion of the
    scalar followed by a sudden renewal (sweep/ejection). This routine uses a
    simple amplitude-threshold conditional sampling scheme:

    1. The series is de-trended by subtracting its block mean so that only the
       fluctuating part ``T'`` is analysed.
    2. A ramp is flagged wherever ``T'`` crosses a threshold of half a standard
       deviation. Falling below ``-0.5*std`` marks a cold sweep and rising above
       ``+0.5*std`` marks a warm ramp; the sign of the first crossing sets the
       ramp direction.
    3. For each detected ramp the amplitude (peak-to-trough range, K) and the
       duration (number of samples divided by the sampling frequency, s) are
       recorded.

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
        detected ramps).
    """
    empty = {"amp": [], "tau": [], "count": 0}

    if T is None or fs is None or fs <= 0:
        return empty

    T = np.asarray(T, dtype=float).ravel()
    if T.size == 0 or np.all(np.isnan(T)):
        return empty

    # Drop NaNs, then require enough samples for a ramp of at least 2/fs seconds.
    T = T[~np.isnan(T)]
    if T.size < 2:
        return empty

    # 1. De-trend by removing the block mean.
    Tp = T - np.mean(T)
    std = np.std(Tp)
    if std == 0:
        return empty

    thr = 0.5 * std
    # Sign of the excursion for each sample: -1 below -thr, +1 above +thr, 0 else.
    state = np.zeros(Tp.size, dtype=int)
    state[Tp < -thr] = -1
    state[Tp > thr] = 1

    amps: list[float] = []
    taus: list[float] = []

    min_samples = 2  # a ramp must span at least 2/fs seconds, i.e. >= 2 samples

    start = None
    sign = 0
    for i, s in enumerate(state):
        if s != 0 and start is None:
            # A new ramp begins on the first threshold crossing.
            start = i
            sign = s
        elif start is not None and (s == 0 or s == -sign):
            # Ramp ends when the signal returns to the neutral band or flips sign.
            seg = Tp[start:i]
            if seg.size >= min_samples:
                amps.append(float(np.ptp(seg)))
                taus.append(seg.size / fs)
            start = None if s == 0 else i
            sign = 0 if s == 0 else s

    # Close a ramp still open at the end of the series.
    if start is not None:
        seg = Tp[start:]
        if seg.size >= min_samples:
            amps.append(float(np.ptp(seg)))
            taus.append(seg.size / fs)

    return {"amp": amps, "tau": taus, "count": len(amps)}