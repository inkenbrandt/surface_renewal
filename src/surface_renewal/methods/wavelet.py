# src/surface_renewal/methods/wavelet.py
r"""Wavelet-based ramp detection (Collineau & Brunet 1993).

Surface-renewal (SR) theory models a scalar (here temperature) time series as a
succession of coherent *ramp* structures: a gradual enrichment/depletion of the
scalar followed by an abrupt renewal (sweep/ejection). Collineau & Brunet (1993)
showed that the continuous wavelet transform (CWT) is a natural tool for
isolating these structures: at the wavelet scale matching the characteristic
ramp duration, the wavelet coefficients change sign across each renewal, so the
zero-crossings of the coefficient series delimit individual events and the scale
of maximum wavelet variance measures the dominant ramp duration.

This module implements just enough of the CWT to do that, using the Mexican-hat
(Ricker) wavelet and plain :mod:`numpy` convolution. It deliberately avoids
``scipy.signal.cwt`` (deprecated and removed in recent SciPy) and does not
depend on PyWavelets.

Relation between wavelet scale and ramp period
----------------------------------------------
The Mexican-hat wavelet is the second derivative of a Gaussian. Its Fourier
transform is :math:`\hat\psi(a\omega) \propto (a\omega)^2 e^{-(a\omega)^2/2}`,
so its **power** spectrum :math:`|\hat\psi(a\omega)|^2 \propto (a\omega)^4
e^{-(a\omega)^2}` peaks, over scale :math:`a`, where :math:`(a\omega)^2 = 2`,
i.e. at :math:`a\,\omega = \sqrt{2}`. A pure sinusoid of angular frequency
:math:`\omega_0 = 2\pi/P` (period :math:`P`) therefore produces maximum wavelet
variance at the scale

.. math::  a_\text{peak} = \frac{\sqrt{2}}{\omega_0}
                        = \frac{\sqrt{2}}{2\pi}\,P,

so the period is recovered from the peak scale by

.. math::  P \;=\; \frac{2\pi}{\sqrt{2}}\,a_\text{peak}
             \;=\; \pi\sqrt{2}\,a_\text{peak} \;\approx\; 4.443\,a_\text{peak}.

We adopt :data:`RICKER_PERIOD_FACTOR` :math:`= 2\pi/\sqrt{2}` as the calibration
factor mapping the peak wavelet scale to the dominant ramp period. (This is the
power-spectrum-peak match; it is close to, but not identical with, Torrence &
Compo's "equivalent Fourier period" :math:`2\pi a/\sqrt{2.5}` for the DOG-2
wavelet, which is derived slightly differently. Either is defensible; we use the
analytically clean power-peak factor and document it here.)

References
----------
Collineau, S., & Brunet, Y. (1993). Detection of turbulent coherent motions in a
    forest canopy. Part II: Time-scales and conditional averages.
    *Boundary-Layer Meteorology*, 66(1-2), 49-73.
Torrence, C., & Compo, G. P. (1998). A practical guide to wavelet analysis.
    *Bulletin of the American Meteorological Society*, 79(1), 61-78.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

# Calibration factor mapping the peak Mexican-hat wavelet scale to the dominant
# ramp period (in the same sample units as the scale). See the module docstring
# for the derivation: it is the location of the wavelet power-spectrum peak,
# a*omega = sqrt(2), inverted to give P = (2*pi/sqrt(2)) * a_peak.
RICKER_PERIOD_FACTOR: float = 2.0 * np.pi / np.sqrt(2.0)  # ~= 4.4429

# Minimum prominence of the wavelet-variance peak (peak / median over scales)
# required to treat the record as containing coherent ramps. A pure sinusoid /
# sawtooth produces a sharp interior peak (ratio >> 1); white noise, whose
# unit-energy wavelet variance is roughly flat across scale, gives a ratio near
# 1. This is the peak-significance test of Collineau & Brunet (1993): below the
# threshold there is no dominant time scale and no flux is reported.
MIN_PEAK_PROMINENCE: float = 3.0


def ricker(points: int, a: float) -> np.ndarray:
    r"""Mexican-hat (Ricker) wavelet sampled on integer offsets about zero.

    .. math::

        \psi(t) = \frac{2}{\sqrt{3a}\,\pi^{1/4}}
                  \left(1 - \left(\tfrac{t}{a}\right)^2\right)
                  \exp\!\left(-\frac{t^2}{2a^2}\right)

    Parameters
    ----------
    points : int
        Number of samples in the returned wavelet. The samples are placed at
        integer offsets centred on zero (``t = -(points-1)/2 ... +(points-1)/2``).
    a : float
        Wavelet scale (width) in samples.

    Returns
    -------
    np.ndarray
        The wavelet, length ``points``. This normalisation gives the wavelet
        unit energy (:math:`\int \psi^2\,dt = 1`) independent of ``a``, so
        wavelet variance is comparable across scales.
    """
    points = int(points)
    if points < 1:
        points = 1
    t = np.arange(points, dtype=float) - (points - 1) / 2.0
    norm = 2.0 / (np.sqrt(3.0 * a) * np.pi ** 0.25)
    ta2 = (t / a) ** 2
    return norm * (1.0 - ta2) * np.exp(-(t ** 2) / (2.0 * a ** 2))


def _fill_nans_linear(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (filled, mask): NaNs linearly interpolated; mask flags originals.

    Leading/trailing NaNs are filled by edge extension (``np.interp`` behaviour).
    ``mask`` is True where the input was NaN, so the transform can be re-masked.
    """
    x = np.asarray(x, dtype=float).ravel()
    mask = np.isnan(x)
    if not mask.any():
        return x.copy(), mask
    idx = np.arange(x.size)
    good = ~mask
    if good.sum() == 0:
        return x.copy(), mask  # all NaN; caller guards on this
    filled = x.copy()
    filled[mask] = np.interp(idx[mask], idx[good], x[good])
    return filled, mask


def cwt_ricker(x: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Continuous wavelet transform of ``x`` with the Ricker wavelet.

    The input is de-meaned first; NaNs are filled by linear interpolation before
    the transform and re-masked (set back to NaN) afterwards. Each row is the
    ``mode="same"`` convolution of the signal with a Ricker wavelet of the
    corresponding scale, using a wavelet length of ``min(10*a, len(x))``.

    Parameters
    ----------
    x : np.ndarray
        Signal (1-D). May contain NaNs.
    scales : np.ndarray
        Wavelet scales in samples.

    Returns
    -------
    np.ndarray
        Coefficient matrix of shape ``(len(scales), len(x))``. Columns
        corresponding to NaN input are set to NaN in every row.
    """
    x = np.asarray(x, dtype=float).ravel()
    scales = np.asarray(scales, dtype=float).ravel()
    n = x.size

    filled, mask = _fill_nans_linear(x)
    filled = filled - np.mean(filled)

    out = np.empty((scales.size, n), dtype=float)
    for i, a in enumerate(scales):
        w_len = int(min(10.0 * a, n))
        if w_len < 1:
            w_len = 1
        psi = ricker(w_len, a)
        out[i] = np.convolve(filled, psi, mode="same")

    if mask.any():
        out[:, mask] = np.nan
    return out


class WaveletRampResult(NamedTuple):
    """Result of wavelet-based ramp detection.

    Attributes
    ----------
    A : float
        Mean ramp amplitude (K, signed). The magnitude is the mean peak-to-peak
        temperature range across detected events; the sign follows the skewness
        of the temperature increments (warming ramps -> positive ``A``).
    tau : float
        Dominant ramp period (s): the mean inter-ramp period (record length /
        number of ramps) when at least three ramps are detected, otherwise the
        scale-based estimate.
    n_ramps : int
        Number of detected ramp events.
    scale_peak_s : float
        Wavelet scale of maximum wavelet variance, expressed as a duration (s).
    H : float
        Uncalibrated sensible heat flux :math:`\\rho c_p A / \\tau` (W m^-2).
    """
    A: float
    tau: float
    n_ramps: int
    scale_peak_s: float
    H: float


def _nan_result() -> WaveletRampResult:
    """Return an all-NaN degenerate result."""
    nan = float("nan")
    return WaveletRampResult(A=nan, tau=nan, n_ramps=0, scale_peak_s=nan, H=nan)


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    """Edge-corrected moving average of ``x`` over a window of ``w`` samples."""
    w = int(w)
    if w <= 1:
        return np.asarray(x, dtype=float)
    kernel = np.ones(w, dtype=float)
    num = np.convolve(x, kernel, mode="same")
    den = np.convolve(np.ones_like(x, dtype=float), kernel, mode="same")
    return num / den


def _skewness(x: np.ndarray) -> float:
    """Sample skewness of a 1-D array (0.0 for degenerate input)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return 0.0
    m = x.mean()
    s = x.std()
    if s == 0.0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def detect_ramps_wavelet(
    T: np.ndarray,
    *,
    hz: float,
    scales_s: tuple[float, float] = (1.0, 120.0),
    n_scales: int = 40,
    rho: float = 1.2,
    cp: float = 1005.0,
) -> WaveletRampResult:
    r"""Detect temperature ramps with the wavelet method of Collineau & Brunet.

    Parameters
    ----------
    T : np.ndarray
        Temperature time series (K). May contain NaNs (filled by interpolation).
    hz : float
        Sampling frequency (Hz).
    scales_s : tuple[float, float], default (1.0, 120.0)
        Lower and upper bounds of the wavelet scales, expressed as durations (s).
    n_scales : int, default 40
        Number of log-spaced scales between the bounds.
    rho : float, default 1.2
        Air density (kg m^-3), used only for the uncalibrated ``H``.
    cp : float, default 1005.0
        Specific heat of air at constant pressure (J kg^-1 K^-1).

    Returns
    -------
    WaveletRampResult
        Mean ramp amplitude, dominant period, event count, peak scale and the
        uncalibrated flux. Degenerate input yields an all-NaN result.

    Notes
    -----
    Algorithm (Collineau & Brunet 1993, Part II):

    a. Build ``n_scales`` log-spaced wavelet scales between ``scales_s`` (in
       samples), transform the de-meaned series, and form the wavelet variance
       :math:`W(a) = \langle |c(a,t)|^2 \rangle_t`.
    b. The scale of maximum variance, :math:`a_\text{peak}`, marks the dominant
       ramp duration; the scale-based period estimate is
       :math:`a_\text{peak} \times` :data:`RICKER_PERIOD_FACTOR`.
    c. At :math:`a_\text{peak}` the coefficient series changes sign across each
       renewal. Ramp events are the lobes between consecutive zero-crossings
       whose extremum exceeds one standard deviation of the coefficients; each
       event's amplitude is the peak-to-peak range of the (despiked) temperature
       within the event window.
    d. ``tau`` is the mean inter-ramp period (record length / ``n_ramps``) when
       ``n_ramps >= 3``, otherwise the scale-based estimate.
    e. ``A`` is the signed mean amplitude, the sign taken from the skewness of
       the temperature increments (warming ramps -> positive ``A``).
    """
    # --- Guards on the basic inputs -------------------------------------------
    if T is None or hz is None or not np.isfinite(hz) or hz <= 0:
        return _nan_result()

    T = np.asarray(T, dtype=float).ravel()
    n = T.size
    if n < 8 or np.all(np.isnan(T)):
        return _nan_result()

    lo_s, hi_s = scales_s
    if not (np.isfinite(lo_s) and np.isfinite(hi_s)) or lo_s <= 0 or hi_s <= lo_s:
        return _nan_result()
    if n_scales < 2:
        return _nan_result()

    # Despiked / gap-filled temperature used for amplitude measurement and the
    # increment skewness. (The pipeline despikes upstream; here we only bridge
    # NaN gaps so the raw ramp amplitudes are well defined.)
    T_clean, _mask = _fill_nans_linear(T)
    if np.nanstd(T_clean) == 0.0:
        return _nan_result()

    # (a) Log-spaced scales in samples, capped so the widest wavelet still fits.
    lo = lo_s * hz
    hi = min(hi_s * hz, float(n))
    if hi <= lo:
        return _nan_result()
    scales = np.logspace(np.log10(lo), np.log10(hi), n_scales)

    coeffs = cwt_ricker(T_clean, scales)

    # (b) Wavelet variance and its peak scale.
    W = np.nanmean(coeffs ** 2, axis=1)
    if not np.any(np.isfinite(W)) or np.nanmax(W) <= 0.0:
        return _nan_result()
    peak_idx = int(np.nanargmax(W))
    a_peak = float(scales[peak_idx])
    scale_peak_s = a_peak / hz
    tau_scale = a_peak * RICKER_PERIOD_FACTOR / hz

    # Peak-significance test: a genuine dominant ramp scale stands out well above
    # the (roughly flat) background variance. White noise has no such peak, so we
    # report the peak scale but no events / flux rather than a spurious value.
    W_med = float(np.nanmedian(W))
    if W_med <= 0.0 or (np.nanmax(W) / W_med) < MIN_PEAK_PROMINENCE:
        return WaveletRampResult(
            A=float("nan"), tau=float(tau_scale), n_ramps=0,
            scale_peak_s=float(scale_peak_s), H=float("nan"),
        )

    # (c) Ramp events. At the peak scale the coefficient series completes one full
    # oscillation (a positive and a negative lobe) per ramp period, so its
    # zero-crossings come in pairs. Each *pair* of consecutive crossings bounds
    # one full ramp; measuring the amplitude and counting over full periods
    # avoids double-counting half-lobes. An event is kept when the coefficient
    # extremum within the window exceeds one standard deviation.
    c = coeffs[peak_idx]
    c = np.where(np.isfinite(c), c, 0.0)
    std_c = float(np.std(c))
    if std_c == 0.0:
        return _nan_result()

    # Ramp amplitude and the increment skewness are read from a lightly denoised
    # ("despiked") temperature: a moving average over a window much shorter than
    # the ramp period suppresses per-sample turbulence noise — which would
    # otherwise inflate a raw max-minus-min range and swamp the sign of dT — while
    # leaving the ramp structure (and its sharp renewal edge) essentially intact.
    w_smooth = max(1, int(round(a_peak / 5.0)))
    T_dsp = _smooth(T_clean, w_smooth)

    sign_c = np.sign(c)
    # Zero-crossing indices: position i where the sign flips between i and i+1.
    crossings = np.where(np.diff(sign_c) != 0)[0]

    amplitudes: list[float] = []
    for k in range(0, len(crossings) - 2, 2):
        start = crossings[k] + 1
        end = crossings[k + 2] + 1  # span two lobes == one full ramp period
        seg = c[start:end]
        if seg.size == 0:
            continue
        if np.max(np.abs(seg)) > std_c:
            amplitudes.append(float(np.ptp(T_dsp[start:end])))

    n_ramps = len(amplitudes)

    # (e) Signed mean amplitude. Sign from the skewness of the increments dT:
    # a warming ramp (gradual rise, sudden drop) has negatively skewed
    # increments and corresponds to an upward heat flux (A, H > 0).
    dT = np.diff(T_dsp)
    skew = _skewness(dT)
    sign = -1.0 if skew > 0.0 else 1.0

    if n_ramps == 0:
        # No coherent events: report the peak scale but no flux.
        return WaveletRampResult(
            A=float("nan"), tau=float(tau_scale), n_ramps=0,
            scale_peak_s=float(scale_peak_s), H=float("nan"),
        )

    A = sign * float(np.mean(amplitudes))

    # (d) Mean inter-ramp period; fall back to the scale estimate for few events.
    record_len_s = n / hz
    if n_ramps >= 3:
        tau = record_len_s / n_ramps
    else:
        tau = tau_scale

    if not np.isfinite(tau) or tau <= 0:
        return _nan_result()

    H = rho * cp * A / tau
    return WaveletRampResult(
        A=float(A), tau=float(tau), n_ramps=int(n_ramps),
        scale_peak_s=float(scale_peak_s), H=float(H),
    )
