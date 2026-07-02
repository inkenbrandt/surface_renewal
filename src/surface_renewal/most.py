# src/surface_renewal/most.py
"""Monin–Obukhov Similarity Theory (MOST) universal functions.

This module collects the dimensionless MOST "universal functions" and the
Monin–Obukhov length that are shared by several surface-flux methods in the
package (e.g. flux–variance and structure-parameter approaches). Keeping them
in one place avoids duplicating the underlying formulae across modules.

All functions are scalar-in, scalar-out and return ``float('nan')`` for
non-finite input so that they can be mapped over arrays with predictable
NaN propagation.

References
----------
Businger, J. A., Wyngaard, J. C., Izumi, Y., & Bradley, E. F. (1971).
    Flux-profile relationships in the atmospheric surface layer.
    *Journal of the Atmospheric Sciences*, 28(2), 181–189.
Dyer, A. J. (1974). A review of flux-profile relationships.
    *Boundary-Layer Meteorology*, 7(3), 363–372.
Tillman, J. E. (1972). The indirect determination of stability, heat and
    momentum fluxes in the atmospheric boundary layer from simple scalar
    variables during dry unstable conditions.
    *Journal of Applied Meteorology*, 11(5), 783–792.
Wyngaard, J. C., Izumi, Y., & Collins, S. A. (1971). Behavior of the
    refractive-index-structure parameter near the ground.
    *Journal of the Optical Society of America*, 61(12), 1646–1650.
"""
from __future__ import annotations

import math

import numpy as np

# Physical constants
KAPPA = 0.41  # von Kármán constant κ (dimensionless)
G = 9.81      # gravitational acceleration g (m s⁻²)


def phi_h(zeta: float) -> float:
    r"""Dimensionless temperature-gradient similarity function :math:`\phi_h`.

    The Businger–Dyer form is used:

    - unstable (:math:`\zeta < 0`):  :math:`\phi_h = (1 - 16\,\zeta)^{-1/2}`
    - stable  (:math:`\zeta \ge 0`):  :math:`\phi_h = 1 + 5\,\zeta`

    Parameters
    ----------
    zeta : float
        Stability parameter :math:`\zeta = z / L` (dimensionless).

    Returns
    -------
    float
        Value of :math:`\phi_h`. Equals ``1.0`` at neutral (``zeta == 0``).
        Returns NaN for non-finite input.

    Notes
    -----
    :math:`\phi_h` scales the mean temperature gradient in the surface layer,
    :math:`\phi_h(\zeta) = \frac{\kappa z}{T_*}\,\partial \bar{T}/\partial z`.

    References
    ----------
    Businger et al. (1971); Dyer (1974). See module docstring for full
    citations.
    """
    if not math.isfinite(zeta):
        return float("nan")
    if zeta < 0.0:
        return float((1.0 - 16.0 * zeta) ** (-0.5))
    return float(1.0 + 5.0 * zeta)


def sigma_T_over_Tstar(zeta: float, *, c1: float = 0.95, c2: float = 0.05) -> float:
    r"""Flux–variance similarity function :math:`\sigma_T / |T_*|`.

    Tillman (1972) form relating the standard deviation of temperature to the
    temperature scale :math:`T_*` as a function of stability:

    - unstable (:math:`\zeta < -c_2`):  :math:`c_1\,(c_2 - \zeta)^{-1/3}`
    - near-neutral / stable (:math:`\zeta \ge -c_2`):  the floor value
      :math:`c_1\,(c_2)^{-1/3}`

    The floor avoids the singularity of the unstable expression as
    :math:`\zeta \to c_2^-` and provides a well-defined value on the
    near-neutral and stable side, where flux–variance similarity for
    temperature is weak.

    Parameters
    ----------
    zeta : float
        Stability parameter :math:`\zeta = z / L` (dimensionless).
    c1 : float, default 0.95
        Similarity coefficient (dimensionless). Site-tunable; the default
        follows Tillman (1972).
    c2 : float, default 0.05
        Offset coefficient (dimensionless) marking the transition to the
        floor value. Site-tunable; the default follows Tillman (1972).

    Returns
    -------
    float
        Value of :math:`\sigma_T / |T_*|`. Returns NaN for non-finite input.

    Notes
    -----
    ``c1`` and ``c2`` are empirical and vary with site, sensor height, and
    surface conditions; they should be recalibrated where possible. The
    defaults reproduce the classic Tillman (1972) dry-unstable relationship.

    References
    ----------
    Tillman, J. E. (1972). See module docstring for the full citation.
    """
    if not math.isfinite(zeta):
        return float("nan")
    if zeta < -c2:
        return float(c1 * (c2 - zeta) ** (-1.0 / 3.0))
    # Near-neutral / stable floor value.
    return float(c1 * c2 ** (-1.0 / 3.0))


def f_CT2(zeta: float, *, c1: float = 4.9, c2: float = 7.0) -> float:
    r"""MOST scaling function for the temperature structure parameter.

    Relates the temperature structure parameter :math:`C_T^2` to stability via

    .. math::

        \frac{C_T^2\, z^{2/3}}{T_*^2} = f_{C_T^2}(\zeta),

    using the Wyngaard et al. (1971) form:

    - unstable (:math:`\zeta < 0`):  :math:`c_1\,(1 - c_2\,\zeta)^{-2/3}`
    - stable  (:math:`\zeta \ge 0`):  :math:`c_1\,(1 + c_2\,\zeta^{2/3})`

    Parameters
    ----------
    zeta : float
        Stability parameter :math:`\zeta = z / L` (dimensionless).
    c1 : float, default 4.9
        Neutral-limit coefficient (dimensionless). Site-tunable; the default
        follows Wyngaard et al. (1971).
    c2 : float, default 7.0
        Stability-dependence coefficient (dimensionless). Site-tunable; the
        default follows Wyngaard et al. (1971).

    Returns
    -------
    float
        Value of :math:`f_{C_T^2}(\zeta)`. Equals ``c1`` at neutral
        (``zeta == 0``). Returns NaN for non-finite input.

    Notes
    -----
    Inverting this relationship allows :math:`T_*` (and hence the sensible heat
    flux) to be recovered from an estimate of :math:`C_T^2`, which the
    surface-renewal structure-function machinery provides.

    References
    ----------
    Wyngaard, J. C., Izumi, Y., & Collins, S. A. (1971). See module docstring
    for the full citation.
    """
    if not math.isfinite(zeta):
        return float("nan")
    if zeta < 0.0:
        return float(c1 * (1.0 - c2 * zeta) ** (-2.0 / 3.0))
    return float(c1 * (1.0 + c2 * zeta ** (2.0 / 3.0)))


def obukhov_length(
    ustar: float,
    T_K: float,
    H: float,
    rho: float = 1.2,
    cp: float = 1005.0,
) -> float:
    r"""Compute the Monin–Obukhov length :math:`L`.

    .. math::

        L = -\frac{\rho\, c_p\, T_K\, u_*^3}{\kappa\, g\, H}

    Parameters
    ----------
    ustar : float
        Friction velocity :math:`u_*` (m s⁻¹).
    T_K : float
        Mean air temperature (K).
    H : float
        Sensible heat flux (W m⁻²).
    rho : float, default 1.2
        Air density (kg m⁻³).
    cp : float, default 1005.0
        Specific heat of air at constant pressure (J kg⁻¹ K⁻¹).

    Returns
    -------
    float
        Monin–Obukhov length :math:`L` (m). Returns NaN if ``H == 0``,
        ``ustar <= 0``, or any input is non-finite.

    Notes
    -----
    ``L`` is negative under unstable (daytime, upward heat flux) conditions and
    positive under stable conditions, consistent with the sign convention used
    by :func:`phi_h` and the other similarity functions in this module.

    References
    ----------
    Businger et al. (1971). See module docstring for the full citation.
    """
    if not all(math.isfinite(x) for x in (ustar, T_K, H, rho, cp)):
        return float("nan")
    if H == 0 or ustar <= 0:
        return float("nan")
    return float(-(rho * cp * T_K * ustar ** 3) / (KAPPA * G * H))
