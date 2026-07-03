import itertools

import numpy as np
import pandas as pd
import pytest

from surface_renewal.pipeline import PipelineConfig
from surface_renewal.methods.analysis import (
    compare_methods,
    method_agreement,
    DEFAULT_COMPARE_METHODS,
)


def _ramp_dataset(
    *, fs: int = 10, minutes: int = 120, period: float = 30.0,
    amp: float = 0.4, seed: int = 0,
) -> pd.DataFrame:
    """Multi-block warming-ramp sawtooth in T with turbulent winds.

    A gradual linear rise then a sharp drop each ``period`` seconds is a
    warming ramp (upward heat flux); the accompanying winds give a well-defined
    friction velocity so the height-dependent methods (fvs, castellvi) run.
    """
    n = fs * 60 * minutes
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)

    phase = (t % period) / period                  # 0 -> 1 sawtooth
    T = 298.15 + amp * (phase - 0.5) + rng.standard_normal(n) * 0.02
    u = 2.5 + rng.standard_normal(n) * 0.5
    v = 0.1 + rng.standard_normal(n) * 0.5
    w = rng.standard_normal(n) * 0.15
    idx = pd.Timestamp("2023-06-01") + pd.to_timedelta(t, unit="s")
    return pd.DataFrame({"T": T, "u": u, "v": v, "w": w}, index=idx)


def test_compare_methods_wide_frame_and_agreement_pairs():
    """All five methods run on one dataset; wide frame + agreement are consistent."""
    df = _ramp_dataset(minutes=120)
    # Relax the u* screen so the synthetic (noise-wind) blocks pass the stability
    # gate; this makes method_agreement compute real slope/rmse/bias, not N=0.
    cfg = PipelineConfig(
        fs=10, block="30min", rotation="double", z_m=2.0, stability_ustar=0.0,
    )

    wide = compare_methods(df, cfg=cfg)
    assert wide["passed"].any()

    # One H column per requested method, each with at least one finite value.
    for m in DEFAULT_COMPARE_METHODS:
        col = f"H_{m}"
        assert col in wide.columns, col
        assert np.isfinite(wide[col].to_numpy(float)).any(), col

    # Shared diagnostics are present.
    for diag in ("ustar", "stdT", "zeta", "CT2", "passed"):
        assert diag in wide.columns, diag

    # zeta is filled by the height-dependent methods.
    assert np.isfinite(wide["zeta"].to_numpy(float)).any()

    # method_agreement returns exactly one row per unordered method pair.
    agree = method_agreement(wide)
    n_methods = len(DEFAULT_COMPARE_METHODS)
    expected_pairs = n_methods * (n_methods - 1) // 2
    assert len(agree) == expected_pairs

    assert list(agree.index.names) == ["method_a", "method_b"]
    assert set(agree.columns) == {"slope", "rmse", "bias", "N"}
    # With passing blocks, at least one pair has real (finite) statistics.
    assert (agree["N"] > 0).any()
    assert np.isfinite(agree.loc[("snyder", "chen97"), "slope"])
    # A method compared against itself is trivially perfect, so it is excluded;
    # every emitted pair is between two distinct methods.
    for a, b in agree.index:
        assert a != b


def test_compare_methods_skips_height_methods_without_zm(caplog):
    """fvs/castellvi are silently skipped (with a warning) when z_m is None."""
    df = _ramp_dataset(minutes=60)
    cfg = PipelineConfig(fs=10, block="30min", rotation="double", z_m=None)

    import logging
    with caplog.at_level(logging.WARNING):
        wide = compare_methods(df, cfg=cfg)

    # Height-dependent methods dropped; calibration-free ones remain.
    assert "H_fvs" not in wide.columns
    assert "H_castellvi" not in wide.columns
    assert "H_snyder" in wide.columns
    assert "H_chen97" in wide.columns
    assert "H_wavelet" in wide.columns

    assert any("z_m" in rec.message for rec in caplog.records)

    # Agreement pairs come only from the three methods that ran.
    agree = method_agreement(wide)
    ran = ["snyder", "chen97", "wavelet"]
    assert len(agree) == len(list(itertools.combinations(ran, 2)))


def test_compare_methods_subset_of_methods():
    """Requesting a subset yields exactly those H columns and their pairs."""
    df = _ramp_dataset(minutes=60)
    cfg = PipelineConfig(fs=10, block="30min", rotation="double")

    wide = compare_methods(df, cfg=cfg, methods=("snyder", "chen97"))
    assert [c for c in wide.columns if c.startswith("H_")] == ["H_snyder", "H_chen97"]

    agree = method_agreement(wide)
    assert len(agree) == 1
    assert ("snyder", "chen97") in agree.index
