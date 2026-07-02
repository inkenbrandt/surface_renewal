import numpy as np
import pytest

from surface_renewal.structure import estimate_CT2


def _fgn_davies_harte(n: int, hurst: float, rng: np.random.Generator) -> np.ndarray:
    """Exact fractional Gaussian noise via the Davies–Harte circulant method.

    The returned fGn has autocovariance
    ``rho(k) = 0.5 * (|k-1|^{2H} - 2|k|^{2H} + |k+1|^{2H})``, i.e. it is the
    increment process of standard fractional Brownian motion normalized so that
    ``Var(B_H(k) - B_H(0)) = k^{2H}``. Cumulatively summing it therefore yields
    a signal whose second-order structure function is ``S2(k) = k^{2H}`` (in
    expectation, with the lag ``k`` measured in samples).
    """
    def rho(k: np.ndarray) -> np.ndarray:
        k = np.asarray(k, float)
        return 0.5 * (
            np.abs(k - 1) ** (2 * hurst)
            - 2 * np.abs(k) ** (2 * hurst)
            + np.abs(k + 1) ** (2 * hurst)
        )

    k = np.arange(0, n + 1)
    row = rho(k)                                   # length n + 1
    circ = np.concatenate([row, row[1:n][::-1]])   # circulant first row, length 2n
    lam = np.clip(np.fft.fft(circ).real, 0, None)  # nonnegative eigenvalues
    m = circ.size
    v = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    w = np.sqrt(lam / m) * v
    return np.fft.fft(w).real[:n]


def test_estimate_CT2_recovers_power_law():
    """With S2(r) ~ c * r^(2/3), estimate_CT2 recovers c within 20%."""
    hz, U = 10.0, 2.0
    rng = np.random.default_rng(0)

    # Hurst = 1/3 gives the 2/3-scaling structure function of the inertial
    # subrange. With unit-variance fGn increments and lag k in samples,
    # S2(k) = k^(2H). Since r = U * k / hz, we have S2 = (hz/U)^(2/3) * r^(2/3),
    # so the true structure parameter is c = (hz/U)^(2/3).
    fgn = _fgn_davies_harte(8192, hurst=1.0 / 3.0, rng=rng)
    signal = np.cumsum(fgn)

    c_true = (hz / U) ** (2.0 / 3.0)
    CT2, r2 = estimate_CT2(signal, hz=hz, U=U)

    assert np.isfinite(CT2)
    assert abs(CT2 - c_true) / c_true < 0.20
    assert r2 > 0.9  # clean power law fits well


def test_estimate_CT2_white_noise_poor_fit():
    """White noise has a flat structure function → poor 2/3-power fit (r2 < 0.5)."""
    rng = np.random.default_rng(1)
    wn = rng.standard_normal(8192)

    CT2, r2 = estimate_CT2(wn, hz=10.0, U=2.0)

    assert np.isfinite(CT2)
    assert r2 < 0.5


def test_estimate_CT2_low_wind_returns_nan():
    """U at or below the near-calm threshold (0.1 m/s) returns (nan, nan)."""
    rng = np.random.default_rng(2)
    sig = np.cumsum(rng.standard_normal(8192))

    assert all(np.isnan(estimate_CT2(sig, hz=10.0, U=0.1)))
    assert all(np.isnan(estimate_CT2(sig, hz=10.0, U=0.05)))


def test_estimate_CT2_too_few_lags_returns_nan():
    """A lag window admitting fewer than 4 distinct lags returns (nan, nan)."""
    rng = np.random.default_rng(3)
    sig = np.cumsum(rng.standard_normal(2048))

    # At 10 Hz, (0.1, 0.3) s -> lags 1..3 (only 3 distinct lags).
    CT2, r2 = estimate_CT2(sig, hz=10.0, U=2.0, lag_range_s=(0.1, 0.3))
    assert np.isnan(CT2) and np.isnan(r2)
