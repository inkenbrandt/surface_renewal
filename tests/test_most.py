from __future__ import annotations

import math

import numpy as np
import pytest

from surface_renewal.most import (
    KAPPA,
    G,
    phi_h,
    sigma_T_over_Tstar,
    f_CT2,
    obukhov_length,
)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
def test_constants():
    """The module constants match the accepted MOST values."""
    assert KAPPA == pytest.approx(0.41)
    assert G == pytest.approx(9.81)


# --------------------------------------------------------------------------- #
# phi_h
# --------------------------------------------------------------------------- #
def test_phi_h_neutral_limit():
    """phi_h(0) == 1 at neutral stability (both branches meet at 1)."""
    assert phi_h(0.0) == pytest.approx(1.0)


def test_phi_h_sign_handling():
    """Unstable and stable branches use the correct expressions."""
    # unstable: (1 - 16*zeta)^(-1/2)
    assert phi_h(-1.0) == pytest.approx((1.0 - 16.0 * (-1.0)) ** (-0.5))
    # stable: 1 + 5*zeta
    assert phi_h(0.5) == pytest.approx(1.0 + 5.0 * 0.5)


def test_phi_h_unstable_below_one():
    """Under unstable conditions phi_h < 1 (enhanced turbulent mixing)."""
    assert phi_h(-0.5) < 1.0


def test_phi_h_stable_monotonic_increasing():
    """phi_h is strictly increasing on the stable branch."""
    zetas = np.linspace(0.0, 5.0, 50)
    vals = np.array([phi_h(z) for z in zetas])
    assert np.all(np.diff(vals) > 0)


def test_phi_h_nan_propagation():
    """Non-finite input yields NaN."""
    assert math.isnan(phi_h(float("nan")))
    assert math.isnan(phi_h(float("inf")))
    assert math.isnan(phi_h(float("-inf")))


# --------------------------------------------------------------------------- #
# sigma_T_over_Tstar
# --------------------------------------------------------------------------- #
def test_sigma_T_unstable():
    """Unstable branch matches c1 * (c2 - zeta)^(-1/3)."""
    c1, c2 = 0.95, 0.05
    zeta = -1.0
    assert sigma_T_over_Tstar(zeta) == pytest.approx(c1 * (c2 - zeta) ** (-1.0 / 3.0))


def test_sigma_T_floor():
    """Near-neutral/stable returns the floor value c1 * c2^(-1/3)."""
    c1, c2 = 0.95, 0.05
    floor = c1 * c2 ** (-1.0 / 3.0)
    # zeta >= -c2 all return the floor
    assert sigma_T_over_Tstar(0.0) == pytest.approx(floor)
    assert sigma_T_over_Tstar(2.0) == pytest.approx(floor)
    assert sigma_T_over_Tstar(-c2 / 2.0) == pytest.approx(floor)


def test_sigma_T_custom_coeffs():
    """c1 and c2 are honoured."""
    assert sigma_T_over_Tstar(-2.0, c1=1.0, c2=0.1) == pytest.approx(
        1.0 * (0.1 - (-2.0)) ** (-1.0 / 3.0)
    )


def test_sigma_T_nan_propagation():
    assert math.isnan(sigma_T_over_Tstar(float("nan")))
    assert math.isnan(sigma_T_over_Tstar(float("inf")))


# --------------------------------------------------------------------------- #
# f_CT2
# --------------------------------------------------------------------------- #
def test_f_CT2_neutral_limit():
    """f_CT2(0) == c1 (both branches equal c1 at neutral)."""
    assert f_CT2(0.0) == pytest.approx(4.9)


def test_f_CT2_sign_handling():
    """Unstable and stable branches use the correct expressions."""
    c1, c2 = 4.9, 7.0
    # unstable
    assert f_CT2(-0.5) == pytest.approx(c1 * (1.0 - c2 * (-0.5)) ** (-2.0 / 3.0))
    # stable
    assert f_CT2(0.5) == pytest.approx(c1 * (1.0 + c2 * 0.5 ** (2.0 / 3.0)))


def test_f_CT2_nan_propagation():
    assert math.isnan(f_CT2(float("nan")))
    assert math.isnan(f_CT2(float("-inf")))


# --------------------------------------------------------------------------- #
# obukhov_length
# --------------------------------------------------------------------------- #
def test_obukhov_length_unstable_negative():
    """Upward (positive) heat flux gives a negative L (unstable)."""
    L = obukhov_length(ustar=0.3, T_K=293.15, H=200.0)
    assert L < 0.0


def test_obukhov_length_stable_positive():
    """Downward (negative) heat flux gives a positive L (stable)."""
    L = obukhov_length(ustar=0.3, T_K=293.15, H=-50.0)
    assert L > 0.0


def test_obukhov_length_formula():
    """L matches the closed-form expression."""
    ustar, T_K, H, rho, cp = 0.3, 293.15, 200.0, 1.2, 1005.0
    expected = -(rho * cp * T_K * ustar ** 3) / (KAPPA * G * H)
    assert obukhov_length(ustar, T_K, H) == pytest.approx(expected)


def test_obukhov_length_degenerate():
    """Returns NaN for H == 0, ustar <= 0, or non-finite input."""
    assert math.isnan(obukhov_length(0.3, 293.15, 0.0))
    assert math.isnan(obukhov_length(0.0, 293.15, 200.0))
    assert math.isnan(obukhov_length(-0.1, 293.15, 200.0))
    assert math.isnan(obukhov_length(float("nan"), 293.15, 200.0))
    assert math.isnan(obukhov_length(0.3, float("inf"), 200.0))


def test_backwards_compatible_reexport():
    """stability.monin_obukhov_length is the same callable as most.obukhov_length."""
    from surface_renewal.preprocess.stability import monin_obukhov_length

    assert monin_obukhov_length is obukhov_length
