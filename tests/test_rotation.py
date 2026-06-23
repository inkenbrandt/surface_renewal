"""Planar-fit rotation regression tests.

Guards the sign of the planar-fit normal: the rotated vertical axis ``k'`` must
point up (positive z-component), and for near-level data the mean rotated
vertical velocity ``w_r`` must be ~0 (the tilt is removed, not doubled).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from surface_renewal.preprocess.rotation import (
    build_planar_fit_matrix,
    planar_fit,
)


def _tilted_winds(n=20_000, b=0.05, c=-0.03, seed=0):
    """Wind with a small fixed tilt plane w = b*u + c*v (no offset) + noise."""
    rng = np.random.default_rng(seed)
    u = 3.0 + 0.5 * rng.standard_normal(n)     # mean wind ~3 m/s along u
    v = 1.0 + 0.5 * rng.standard_normal(n)
    w = b * u + c * v + 0.02 * rng.standard_normal(n)
    return pd.DataFrame({"u": u, "v": v, "w": w})


def test_planar_fit_vertical_axis_points_up():
    df = _tilted_winds()
    R, meta = build_planar_fit_matrix(
        df["u"].to_numpy(), df["v"].to_numpy(), df["w"].to_numpy()
    )
    k_prime = meta["k_prime"]
    # The vertical axis (third row of R == k') must have a positive z-component.
    assert k_prime[2] > 0.0
    assert R[2, 2] > 0.0


def test_planar_fit_zeros_mean_w_for_near_level_data():
    df = _tilted_winds()
    res = planar_fit(df)
    w_r = res.df["w_r"].to_numpy()
    # Tilt removed: mean vertical velocity in the rotated frame is ~0.
    assert abs(np.nanmean(w_r)) < 1e-3
    # And it is genuinely smaller than the raw (tilted) mean w.
    assert abs(np.nanmean(w_r)) < abs(np.nanmean(df["w"].to_numpy()))
