import math

import numpy as np
import pandas as pd
import pytest

from surface_renewal.most import G, obukhov_length, sigma_T_over_Tstar
from surface_renewal.methods.fvs import (
    estimate_H_fvs,
    estimate_H_free_convection,
    FVSResult,
)
from surface_renewal.pipeline import PipelineConfig, run_surface_renewal


def test_fvs_round_trip():
    """A sigma_T generated from a known H must be inverted back to that H."""
    H_true = 200.0
    ustar = 0.4
    T_K = 300.0
    z_m = 3.0
    rho, cp = 1.2, 1005.0

    # Forward: known H -> T*, L, zeta -> sigma_T consistent with similarity.
    abs_Tstar = abs(-H_true / (rho * cp * ustar))
    L = obukhov_length(ustar, T_K, H_true, rho=rho, cp=cp)
    zeta = z_m / L
    assert zeta < 0.0  # unstable, as required by the FVS relation
    sigma_T = abs_Tstar * sigma_T_over_Tstar(zeta)

    # Inverse: recover H from sigma_T (sign supplied via sign_hint > 0 -> H > 0).
    res = estimate_H_fvs(
        sigma_T=sigma_T, ustar=ustar, T_K=T_K, z_m=z_m,
        rho=rho, cp=cp, sign_hint=1.0,
    )

    assert res.converged
    assert res.H == pytest.approx(H_true, rel=0.01)
    assert res.zeta == pytest.approx(zeta, rel=0.01)


def test_fvs_sign_hint_sets_direction():
    """The sign of H (and T*) follows sign_hint; H and T* have opposite signs."""
    common = dict(sigma_T=0.5, ustar=0.4, T_K=300.0, z_m=3.0)
    pos = estimate_H_fvs(sign_hint=1.0, **common)
    neg = estimate_H_fvs(sign_hint=-1.0, **common)
    # T* = -H / (rho cp ustar), so H and T* always carry opposite signs.
    assert pos.H > 0 and pos.Tstar < 0
    assert neg.H < 0 and neg.Tstar > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(sigma_T=np.nan, ustar=0.4, T_K=300.0, z_m=3.0),   # non-finite sigma_T
        dict(sigma_T=0.5, ustar=np.nan, T_K=300.0, z_m=3.0),   # non-finite ustar
        dict(sigma_T=0.5, ustar=0.4, T_K=np.inf, z_m=3.0),     # non-finite T_K
        dict(sigma_T=0.5, ustar=0.005, T_K=300.0, z_m=3.0),    # ustar < 0.01
    ],
)
def test_fvs_degenerate_inputs_return_nan(kwargs):
    """Degenerate inputs yield an all-NaN, non-converged result."""
    res = estimate_H_fvs(**kwargs)
    assert isinstance(res, FVSResult)
    assert not res.converged
    assert np.isnan(res.H)
    assert np.isnan(res.Tstar)
    assert np.isnan(res.zeta)


def test_free_convection_closed_form():
    """The free-convection formula matches its hand-computed closed form."""
    sigma_T, T_K, z_m = 0.5, 300.0, 3.0
    rho, cp, c_fc = 1.2, 1005.0, 0.9

    # H_fc = rho * cp * c_fc * sigma_T**1.5 * sqrt(G * z_m / T_K)
    expected = rho * cp * c_fc * sigma_T ** 1.5 * math.sqrt(G * z_m / T_K)
    got = estimate_H_free_convection(sigma_T=sigma_T, T_K=T_K, z_m=z_m)

    assert got == pytest.approx(expected, abs=1e-6)
    assert got > 0.0  # returns |H| for the unstable case, positive by construction


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(sigma_T=np.nan, T_K=300.0, z_m=3.0),   # non-finite sigma_T
        dict(sigma_T=0.5, T_K=np.inf, z_m=3.0),      # non-finite T_K
        dict(sigma_T=-0.5, T_K=300.0, z_m=3.0),      # negative sigma_T
        dict(sigma_T=0.5, T_K=0.0, z_m=3.0),         # non-physical T_K
        dict(sigma_T=0.5, T_K=300.0, z_m=0.0),       # non-physical z_m
    ],
)
def test_free_convection_degenerate_inputs_nan(kwargs):
    """Non-finite / non-physical inputs return NaN."""
    assert np.isnan(estimate_H_free_convection(**kwargs))


def _low_wind_segment(fs: int = 10, sign: float = -1.0, wind_scale: float = 0.3) -> pd.DataFrame:
    """Strongly unstable, low-wind block: ramp temperature, tiny wind noise.

    ``sign = -1`` shapes the sawtooth so the flux-variance sign comes out
    positive (upward flux), which is the case the free-convection fallback is
    allowed to substitute. Scaling the wind noise way down keeps ``u*`` small
    enough to trip the low-wind fallback threshold.
    """
    rng = np.random.default_rng(0)
    n_rows = fs * 60 * 30
    t = np.arange(n_rows) / fs
    phase = (t % 60.0) / 60.0
    ramp = sign * (phase - 0.5)
    T = 298.15 + 0.5 * ramp + rng.standard_normal(n_rows) * 0.02
    u = 0.5 + rng.standard_normal(n_rows) * wind_scale
    v = 0.1 + rng.standard_normal(n_rows) * wind_scale
    w = rng.standard_normal(n_rows) * wind_scale
    start = pd.Timestamp("2023-01-01 00:00:00")
    index = start + pd.to_timedelta(t, unit="s")
    return pd.DataFrame({"T": T, "u": u, "v": v, "w": w}, index=index)


def test_free_convection_fallback_engages_on_low_wind():
    """On a strongly unstable, low-wind block the fallback substitutes H."""
    df = _low_wind_segment()

    base_cfg = PipelineConfig(fs=10, method="fvs", rotation="double",
                              block="30min", z_m=3.0)
    fb_cfg = PipelineConfig(fs=10, method="fvs", rotation="double",
                            block="30min", z_m=3.0, free_convection_fallback=True)

    base = run_surface_renewal(df, cfg=base_cfg)
    fb = run_surface_renewal(df, cfg=fb_cfg)

    # Sanity: the synthetic block is exactly the low-wind, unstable, upward-flux
    # regime the fallback targets.
    assert (base["ustar"] < fb_cfg.fc_ustar_max).all()
    assert (base["zeta"] < fb_cfg.fc_zeta_max).all()
    assert (base["H_uncal"] > 0).all()

    # Without the flag: primary everywhere. With it: this block switches over.
    assert (base["flux_method_used"] == "primary").all()
    assert (fb["flux_method_used"] == "free_convection").all()

    # The reported H matches the closed-form free-convection estimate.
    stdT = float(fb["stdT"].iloc[0])
    expected = estimate_H_free_convection(
        sigma_T=stdT, T_K=298.15, z_m=3.0,   # block-mean T ≈ 298.15 K
        rho=float(fb["rho"].iloc[0]), cp=float(fb["cp"].iloc[0]),
    )
    assert float(fb["H_uncal"].iloc[0]) == pytest.approx(expected, rel=0.05)


def test_free_convection_fallback_keeps_negative_primary():
    """A negative primary H (downward flux) is never replaced by free convection."""
    # sign=+1 shapes the ramp so the flux-variance sign comes out negative.
    df = _low_wind_segment(sign=1.0)
    fb_cfg = PipelineConfig(fs=10, method="fvs", rotation="double",
                            block="30min", z_m=3.0, free_convection_fallback=True)
    fb = run_surface_renewal(df, cfg=fb_cfg)

    # Primary H is negative here, so the (positive-only) free-convection estimate
    # must not be substituted.
    assert (fb["H_uncal"] < 0).all()
    assert (fb["flux_method_used"] == "primary").all()


def _fvs_segment(fs: int = 10) -> pd.DataFrame:
    """Build a synthetic 30-minute, DatetimeIndex-indexed segment."""
    rng = np.random.default_rng(0)
    n_rows = fs * 60 * 30
    t = np.arange(n_rows) / fs
    T = 298.15 + 0.3 * np.sin(2 * np.pi * t / 60.0) + rng.standard_normal(n_rows) * 0.02
    u = 2.5 + rng.standard_normal(n_rows) * 0.5
    v = 0.1 + rng.standard_normal(n_rows) * 0.5
    w = rng.standard_normal(n_rows) * 0.15
    start = pd.Timestamp("2023-01-01 00:00:00")
    index = start + pd.to_timedelta(t, unit="s")
    return pd.DataFrame({"T": T, "u": u, "v": v, "w": w}, index=index)


def test_fvs_pipeline_runs():
    """The FVS pipeline runs end-to-end with a configured measurement height."""
    df = _fvs_segment(fs=10)
    config = PipelineConfig(fs=10, method="fvs", rotation="double", block="30min", z_m=3.0)
    result = run_surface_renewal(df, cfg=config)

    assert len(result) >= 1
    assert "H_uncal" in result.columns
    assert "zeta" in result.columns


def test_fvs_pipeline_requires_z_m():
    """FVS without z_m raises a clear ValueError."""
    df = _fvs_segment(fs=10)
    config = PipelineConfig(fs=10, method="fvs", rotation="double", block="30min", z_m=None)
    with pytest.raises(ValueError, match="requires cfg.z_m"):
        run_surface_renewal(df, cfg=config)
