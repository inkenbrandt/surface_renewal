import numpy as np
import pandas as pd
import pytest

from surface_renewal.methods.castellvi import (
    CastellviResult,
    alpha_castellvi,
    alpha_from_CT2,
    estimate_H_castellvi,
)
from surface_renewal.most import f_CT2
from surface_renewal.pipeline import PipelineConfig, run_surface_renewal


def _ramp_series(fs: int = 10, minutes: int = 30, amp: float = 0.4,
                 period_s: float = 30.0, sign: float = 1.0,
                 seed: int = 0) -> np.ndarray:
    """Sawtooth (asymmetric ramp) temperature series + small noise.

    A sawtooth has a slow ramp and a sharp drop, the canonical surface-renewal
    signal, so the Van Atta cubic recovery returns a finite (A, tau). ``sign``
    flips the ramp direction: positive -> warming ramps (H > 0).
    """
    rng = np.random.default_rng(seed)
    n = fs * 60 * minutes
    t = np.arange(n) / fs
    # scipy-free sawtooth in [-1, 1]: rising ramp over each period.
    frac = (t % period_s) / period_s
    saw = 2.0 * frac - 1.0
    T = 298.15 + sign * amp * saw + rng.standard_normal(n) * 0.01
    return T


def test_castellvi_finite_and_plausible_alpha():
    """A ramp-like series yields finite H, converged, alpha in a plausible range."""
    T = _ramp_series(fs=10, period_s=30.0)
    res = estimate_H_castellvi(
        T, hz=10.0, ustar=0.3, T_K=298.15, z_m=3.0,
    )
    assert isinstance(res, CastellviResult)
    assert res.converged
    assert np.isfinite(res.H)
    assert np.isfinite(res.A) and np.isfinite(res.tau) and res.tau > 0
    assert 0.1 <= res.alpha <= 2.0


def test_castellvi_sign_follows_ramp():
    """H is signed by the recovered ramp amplitude A (no external sign hint).

    Since ``H = rho*cp*alpha*(A/tau)`` with ``alpha, tau, rho, cp > 0``, the sign
    of H must equal the sign of A. A warming ramp yields the daytime unstable
    case Castellví (2004) targets: A > 0 and H > 0.
    """
    warm = estimate_H_castellvi(_ramp_series(sign=+1.0), hz=10.0, ustar=0.3,
                                T_K=298.15, z_m=3.0)
    assert warm.converged
    assert warm.A > 0.0 and warm.H > 0.0
    assert np.sign(warm.H) == np.sign(warm.A)


def test_alpha_decreases_with_tau():
    """The analytic alpha ~ 1/sqrt(tau): larger tau -> smaller alpha."""
    common = dict(z_m=3.0, ustar=0.3, zeta=0.0)
    a_small = alpha_castellvi(tau=10.0, **common)
    a_mid = alpha_castellvi(tau=30.0, **common)
    a_large = alpha_castellvi(tau=90.0, **common)
    assert a_small > a_mid > a_large
    # Exact 1/sqrt(tau) scaling at fixed stability.
    assert a_small / a_large == pytest.approx(np.sqrt(90.0 / 10.0), rel=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(ustar=np.nan, T_K=298.15, z_m=3.0),   # non-finite ustar
        dict(ustar=0.3, T_K=np.inf, z_m=3.0),      # non-finite T_K
        dict(ustar=0.005, T_K=298.15, z_m=3.0),    # ustar too small
        dict(ustar=0.3, T_K=298.15, z_m=-1.0),     # non-physical height
    ],
)
def test_castellvi_degenerate_inputs_return_nan(kwargs):
    """Degenerate inputs yield an all-NaN, non-converged result."""
    T = _ramp_series()
    res = estimate_H_castellvi(T, hz=10.0, **kwargs)
    assert isinstance(res, CastellviResult)
    assert not res.converged
    assert np.isnan(res.H)
    assert np.isnan(res.alpha)


def test_alpha_from_CT2_cross_check():
    """alpha_from_CT2 inverts the f_CT2 relation for T* at a given stability."""
    z_m, zeta = 3.0, -0.5
    Tstar_true = 0.25
    # Forward: build CT2 consistent with T*_true, then invert.
    CT2 = Tstar_true ** 2 * f_CT2(zeta) / z_m ** (2.0 / 3.0)
    Tstar_est = alpha_from_CT2(CT2, z_m=z_m, Tstar=Tstar_true, zeta=zeta)
    assert Tstar_est == pytest.approx(Tstar_true, rel=1e-9)
    # Sign follows the reference Tstar.
    assert alpha_from_CT2(CT2, z_m=z_m, Tstar=-1.0, zeta=zeta) < 0.0


def _castellvi_segment(fs: int = 10) -> pd.DataFrame:
    """Synthetic 30-minute segment with a ramp-like T and realistic winds."""
    rng = np.random.default_rng(1)
    n = fs * 60 * 30
    t = np.arange(n) / fs
    frac = (t % 30.0) / 30.0
    T = 298.15 + 0.4 * (2.0 * frac - 1.0) + rng.standard_normal(n) * 0.01
    u = 2.5 + rng.standard_normal(n) * 0.5 + 0.5 * np.sin(2 * np.pi * t / 60)
    v = 0.1 + rng.standard_normal(n) * 0.5
    w = rng.standard_normal(n) * 0.15
    start = pd.Timestamp("2023-01-01 00:00:00")
    index = start + pd.to_timedelta(t, unit="s")
    return pd.DataFrame({"T": T, "u": u, "v": v, "w": w}, index=index)


def test_castellvi_pipeline_runs():
    """The Castellví pipeline runs end-to-end and reports alpha_sr."""
    df = _castellvi_segment(fs=10)
    cfg = PipelineConfig(fs=10, method="castellvi", rotation="double",
                         block="30min", z_m=3.0)
    result = run_surface_renewal(df, cfg=cfg)

    assert len(result) >= 1
    assert "H_uncal" in result.columns
    assert "zeta" in result.columns
    assert "alpha_sr" in result.columns
    assert np.isfinite(result["alpha_sr"]).any()


def test_castellvi_pipeline_requires_z_m():
    """Castellví without z_m raises a clear ValueError."""
    df = _castellvi_segment(fs=10)
    cfg = PipelineConfig(fs=10, method="castellvi", rotation="double",
                         block="30min", z_m=None)
    with pytest.raises(ValueError, match="requires cfg.z_m"):
        run_surface_renewal(df, cfg=cfg)
