from __future__ import annotations
import numpy as np
import pytest
from surface_renewal.preprocess.stability import BlockDiagnostics, stability_ok

@pytest.fixture
def good_diag() -> BlockDiagnostics:
    """Return a default BlockDiagnostics that should pass a screen."""
    return BlockDiagnostics(u_star=0.2, S3_tau=-0.005, tau_opt=1.2, stdT=0.1)

def test_stability_ok_pure(good_diag):
    """Check that stability_ok is a pure function (no side effects)."""
    # Test that stability_ok returns True for good diagnostics, and that it
    # does NOT modify the passed-in object.
    res = stability_ok(good_diag)
    assert res is True
    assert good_diag.passed is None  # unchanged

def test_stability_ok_dict_input():
    """Check that stability_ok works with dicts too."""
    diag_dict = {"u_star": 0.2, "S3_tau": -0.005, "tau_opt": 1.2, "stdT": 0.1}
    res = stability_ok(diag_dict)
    assert res is True
    assert "passed" not in diag_dict

def test_stability_screen_thresholds(good_diag):
    """Check each threshold failure case."""
    # Low u*
    assert not stability_ok(good_diag, min_ustar=0.3)
    # Low stdT
    assert not stability_ok(good_diag, min_stdT=0.2)
    # Low S3
    assert not stability_ok(good_diag, min_rel_S3=999)
    # Daytime filter
    assert not stability_ok(good_diag, daytime_only=True, Rn_block=-10)
    assert stability_ok(good_diag, daytime_only=True, Rn_block=10)

def test_stability_nan_cases(good_diag):
    """Check for correct handling of NaN inputs."""
    good_diag.u_star = np.nan
    assert not stability_ok(good_diag)
