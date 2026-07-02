import numpy as np
import pandas as pd
import pytest

from surface_renewal.most import obukhov_length, sigma_T_over_Tstar
from surface_renewal.methods.fvs import estimate_H_fvs, FVSResult
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
