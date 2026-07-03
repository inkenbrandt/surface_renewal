import numpy as np
import pandas as pd
import pytest

from surface_renewal.methods.wavelet import (
    ricker,
    cwt_ricker,
    detect_ramps_wavelet,
    WaveletRampResult,
    RICKER_PERIOD_FACTOR,
)
from surface_renewal.methods.analysis import detect_ramps
from surface_renewal.pipeline import PipelineConfig, run_surface_renewal


# --------------------------------------------------------------------------- #
# Synthetic signal builders
# --------------------------------------------------------------------------- #
def _sawtooth(
    *, period: float = 30.0, amp: float = 0.4, hz: float = 10.0,
    dur: float = 300.0, noise: float = 0.0, seed: int = 0, base: float = 298.15,
) -> np.ndarray:
    """Warming-ramp sawtooth: gradual linear rise then sudden drop each period.

    The peak-to-peak amplitude is ``amp`` (K) and the period is ``period`` (s).
    A gradual rise followed by a sharp drop is a *warming* ramp (upward heat
    flux), so the recovered signed amplitude should be positive.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(dur * hz))) / hz
    phase = (t % period) / period          # 0 -> 1 sawtooth
    T = base + amp * (phase - 0.5)
    if noise > 0.0:
        T = T + rng.standard_normal(t.size) * noise
    return T


# --------------------------------------------------------------------------- #
# ricker / cwt building blocks
# --------------------------------------------------------------------------- #
def test_ricker_is_symmetric_zero_mean_unit_energy():
    """The Ricker wavelet is symmetric, (near) zero-mean and unit-energy."""
    psi = ricker(400, 20.0)
    assert psi.shape == (400,)
    # Symmetric about the centre.
    assert np.allclose(psi, psi[::-1], atol=1e-12)
    # Second derivative of a Gaussian integrates to ~0.
    assert abs(psi.sum()) < 1e-6
    # Normalisation gives unit energy (integral of psi**2 == 1) for wide support.
    assert psi.dot(psi) == pytest.approx(1.0, rel=0.02)


def test_ricker_peaks_at_centre():
    """The wavelet has its positive maximum at the centre sample."""
    psi = ricker(201, 15.0)
    assert np.argmax(psi) == 100  # centre of 201 samples


def test_cwt_shape_and_nan_handling():
    """CWT returns (n_scales, n) and re-masks NaN columns of the input."""
    x = np.sin(np.arange(500) * 0.1)
    x[100] = np.nan
    x[250:253] = np.nan
    scales = np.array([5.0, 10.0, 20.0])
    C = cwt_ricker(x, scales)
    assert C.shape == (3, 500)
    # NaN input columns stay NaN in every row; the rest are finite.
    assert np.all(np.isnan(C[:, 100]))
    assert np.all(np.isnan(C[:, 250:253]))
    assert np.all(np.isfinite(C[:, 0]))


def test_cwt_demeans_input():
    """A large DC offset does not change the coefficients (input is de-meaned)."""
    x = np.sin(np.arange(400) * 0.05)
    scales = np.array([8.0, 16.0])
    C0 = cwt_ricker(x, scales)
    C1 = cwt_ricker(x + 1000.0, scales)
    assert np.allclose(C0, C1, atol=1e-9)


# --------------------------------------------------------------------------- #
# detect_ramps_wavelet — the core acceptance tests
# --------------------------------------------------------------------------- #
def test_pure_sawtooth_recovers_period_amplitude_count():
    """Pure 30 s / 0.4 K sawtooth at 10 Hz: tau, A and count within tolerance."""
    hz, period, amp, dur = 10.0, 30.0, 0.4, 300.0
    true_count = int(dur / period)  # 10 ramps

    res = detect_ramps_wavelet(_sawtooth(period=period, amp=amp, hz=hz, dur=dur), hz=hz)

    assert isinstance(res, WaveletRampResult)
    assert res.tau == pytest.approx(period, rel=0.25)       # within 25 %
    assert res.A == pytest.approx(amp, rel=0.30)            # within 30 %, signed +
    assert res.A > 0.0                                      # warming ramp
    assert abs(res.n_ramps - true_count) <= 2               # within +/-2
    assert res.H > 0.0 and np.isfinite(res.H)


def test_sawtooth_plus_noise_still_within_tolerance():
    """Sawtooth + white noise at SNR ~ 3 stays within the same tolerances."""
    hz, period, amp, dur = 10.0, 30.0, 0.4, 300.0
    true_count = int(dur / period)
    # SNR ~ 3 defined on RMS: sawtooth RMS = amp / sqrt(12); noise RMS = that / 3.
    noise = (amp / np.sqrt(12.0)) / 3.0

    for seed in range(4):
        res = detect_ramps_wavelet(
            _sawtooth(period=period, amp=amp, hz=hz, dur=dur, noise=noise, seed=seed),
            hz=hz,
        )
        assert res.tau == pytest.approx(period, rel=0.25), seed
        assert res.A == pytest.approx(amp, rel=0.30), seed
        assert res.A > 0.0, seed
        assert abs(res.n_ramps - true_count) <= 2, seed


def test_white_noise_no_spurious_flux():
    """Pure white noise: no dominant scale -> few/no ramps and no large H."""
    hz = 10.0
    for seed in range(5):
        rng = np.random.default_rng(seed)
        T = 298.15 + rng.standard_normal(3000) * 0.1
        res = detect_ramps_wavelet(T, hz=hz)
        # Either NaN (peak-significance test rejected the record) or a small
        # count -- but never a confident, physically large flux.
        assert np.isnan(res.H) or res.n_ramps <= 3, seed
        assert not (np.isfinite(res.H) and abs(res.H) > 10.0), seed


def test_cooling_ramp_gives_negative_amplitude():
    """A cooling ramp (sudden rise, gradual fall) yields a negative signed A."""
    hz, period, amp, dur = 10.0, 30.0, 0.4, 300.0
    # Negating the warming sawtooth makes a gradual fall with a sharp rise.
    T = -_sawtooth(period=period, amp=amp, hz=hz, dur=dur, base=0.0) + 298.15
    res = detect_ramps_wavelet(T, hz=hz)
    assert res.A < 0.0
    assert res.H < 0.0


@pytest.mark.parametrize(
    "T, hz",
    [
        (np.array([]), 10.0),                       # empty
        (np.full(500, np.nan), 10.0),               # all NaN
        (np.ones(500), 10.0),                       # constant (zero variance)
        (np.arange(4.0), 10.0),                     # too short
        (np.random.default_rng(0).standard_normal(500), 0.0),   # hz <= 0
    ],
)
def test_degenerate_inputs_return_nan(T, hz):
    """Degenerate inputs yield an all-NaN, zero-ramp result."""
    res = detect_ramps_wavelet(T, hz=hz)
    assert isinstance(res, WaveletRampResult)
    assert np.isnan(res.H)
    assert res.n_ramps == 0


# --------------------------------------------------------------------------- #
# analysis.detect_ramps deprecation shim
# --------------------------------------------------------------------------- #
def test_detect_ramps_deprecated_delegates():
    """The legacy detect_ramps warns and delegates to the wavelet detector."""
    T = _sawtooth()
    with pytest.warns(DeprecationWarning):
        out = detect_ramps(T, 10)
    assert set(out) == {"amp", "tau", "count"}
    assert out["count"] > 0
    assert out["amp"][0] == pytest.approx(0.4, rel=0.30)


# --------------------------------------------------------------------------- #
# Pipeline wiring
# --------------------------------------------------------------------------- #
def _wavelet_segment(fs: int = 10) -> pd.DataFrame:
    """A 30-minute block whose temperature is a 30 s warming-ramp sawtooth."""
    n_rows = fs * 60 * 30
    t = np.arange(n_rows) / fs
    rng = np.random.default_rng(0)
    phase = (t % 30.0) / 30.0
    T = 298.15 + 0.4 * (phase - 0.5) + rng.standard_normal(n_rows) * 0.02
    u = 2.5 + rng.standard_normal(n_rows) * 0.5
    v = 0.1 + rng.standard_normal(n_rows) * 0.5
    w = rng.standard_normal(n_rows) * 0.15
    idx = pd.Timestamp("2023-01-01") + pd.to_timedelta(t, unit="s")
    return pd.DataFrame({"T": T, "u": u, "v": v, "w": w}, index=idx)


def test_wavelet_pipeline_produces_n_ramps_column():
    """The wavelet method runs end-to-end and fills H_uncal, tau_star, n_ramps."""
    df = _wavelet_segment(fs=10)
    cfg = PipelineConfig(fs=10, method="wavelet", rotation="double", block="30min")
    out = run_surface_renewal(df, cfg=cfg)

    assert len(out) >= 1
    assert "n_ramps" in out.columns
    row = out.iloc[0]
    assert row["n_ramps"] > 0
    assert np.isfinite(row["H_uncal"])
    assert np.isfinite(row["tau_star"])
    assert row["tau_star"] == pytest.approx(30.0, rel=0.25)
