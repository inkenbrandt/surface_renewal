# src/surface_renewal/methods/fvs.py
"""Flux–variance similarity (FVS) sensible heat flux estimation.

The flux–variance method infers the sensible heat flux from the standard
deviation of temperature using Monin–Obukhov similarity theory. Under unstable
conditions the temperature standard deviation scales with the temperature
scale :math:`T_*` as (Tillman 1972; Katul et al. 1995)

.. math::

    \\frac{\\sigma_T}{|T_*|} = c_1\\,(c_2 - z/L)^{-1/3},

with :math:`T_* = -H / (\\rho c_p u_*)` and :math:`L` the Obukhov length.
Because :math:`L` itself depends on :math:`H`, the relationship is solved
iteratively: an initial guess of :math:`T_*` seeds :math:`L`, which yields
:math:`\\zeta = z/L`, which updates :math:`|T_*|` through the inverse of the
similarity function, and so on until :math:`H` converges.

The similarity function :math:`\\sigma_T / |T_*|` and the Obukhov length are
imported from :mod:`surface_renewal.most` rather than re-implemented here, so
that all methods share a single definition of the MOST relations.

References
----------
Tillman, J. E. (1972). The indirect determination of stability, heat and
    momentum fluxes in the atmospheric boundary layer from simple scalar
    variables during dry unstable conditions.
    *Journal of Applied Meteorology*, 11(5), 783–792.
Katul, G. G., Goltz, S. M., Hsieh, C.-I., Cheng, Y., Mowry, F., & Sigmon, J.
    (1995). Estimation of surface heat and momentum fluxes using the
    flux-variance method above uniform and non-uniform terrain.
    *Boundary-Layer Meteorology*, 74(3), 237–260.
"""
from __future__ import annotations

import math
from typing import NamedTuple

from ..most import obukhov_length, sigma_T_over_Tstar


class FVSResult(NamedTuple):
    """Result of a flux–variance similarity sensible-heat estimate.

    Attributes
    ----------
    H : float
        Sensible heat flux (W m⁻²). The magnitude comes from flux–variance
        similarity; the sign is supplied by ``sign_hint`` (see
        :func:`estimate_H_fvs`).
    Tstar : float
        Temperature scale :math:`T_* = -H / (\\rho c_p u_*)` (K). Because of the
        leading minus sign it carries the opposite sign to ``H``.
    zeta : float
        Stability parameter :math:`\\zeta = z / L` at convergence.
    n_iter : int
        Number of iterations performed.
    converged : bool
        Whether the iteration met the convergence tolerance.
    """
    H: float
    Tstar: float
    zeta: float
    n_iter: int
    converged: bool


def _nan_result(n_iter: int = 0) -> FVSResult:
    """Return an all-NaN, non-converged result."""
    nan = float("nan")
    return FVSResult(H=nan, Tstar=nan, zeta=nan, n_iter=n_iter, converged=False)


def estimate_H_fvs(
    *,
    sigma_T: float,
    ustar: float,
    T_K: float,
    z_m: float,
    rho: float = 1.2,
    cp: float = 1005.0,
    sign_hint: float = 1.0,
    max_iter: int = 20,
    tol: float = 1e-3,
) -> FVSResult:
    r"""Estimate sensible heat flux from temperature variance via FVS.

    Solves the flux–variance similarity relation

    .. math::

        \frac{\sigma_T}{|T_*|} = c_1\,(c_2 - z/L)^{-1/3}

    iteratively for :math:`H`. Each iteration recomputes the Obukhov length
    from the current :math:`H`, derives :math:`\zeta = z/L`, inverts the
    similarity function to update :math:`|T_*| = \sigma_T /
    (\sigma_T/|T_*|)(\zeta)`, and forms a new
    :math:`H = \mathrm{sign\_hint}\cdot \rho c_p u_* |T_*|`.

    Parameters
    ----------
    sigma_T : float
        Standard deviation of temperature over the averaging block (K).
    ustar : float
        Friction velocity :math:`u_*` (m s⁻¹).
    T_K : float
        Block-mean air temperature (K).
    z_m : float
        Measurement height above the zero-plane displacement (m).
    rho : float, default 1.2
        Air density (kg m⁻³).
    cp : float, default 1005.0
        Specific heat of air at constant pressure (J kg⁻¹ K⁻¹).
    sign_hint : float, default 1.0
        Sign (±1) applied to the recovered magnitude of :math:`H`.
        Flux–variance similarity yields only :math:`|H|`; the direction of the
        heat flux must be supplied from an independent source. Callers should
        pass ``sign_hint = sign(S3(tau*))``: a negative third-order structure
        function indicates warming ramps (daytime, upward flux, :math:`H > 0`).
    max_iter : int, default 20
        Maximum number of fixed-point iterations.
    tol : float, default 1e-3
        Convergence tolerance on successive :math:`H` values (W m⁻²).

    Returns
    -------
    FVSResult
        The estimated flux, temperature scale, stability parameter, iteration
        count, and convergence flag. A NaN-filled, non-converged result is
        returned when ``sigma_T``, ``ustar``, or ``T_K`` is non-finite, or when
        ``ustar < 0.01`` (too weak for a reliable similarity inversion).

    Notes
    -----
    ``sign_hint`` sets the sign of ``H``. ``Tstar`` is then formed as
    :math:`T_* = -H / (\rho c_p u_*)`, so it carries the opposite sign to ``H``.
    """
    # Guard against degenerate / non-physical inputs before iterating.
    if not (math.isfinite(sigma_T) and math.isfinite(ustar) and math.isfinite(T_K)):
        return _nan_result()
    if ustar < 0.01:
        return _nan_result()

    sign = 1.0 if sign_hint >= 0.0 else -1.0

    # (a) Initial guess: crude |T*| ~ sigma_T / 2, seeding an initial H.
    abs_Tstar = sigma_T / 2.0
    H = sign * rho * cp * ustar * abs_Tstar

    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        # (b) Update L, zeta, |T*| and H from the current estimate.
        L = obukhov_length(ustar, T_K, H, rho=rho, cp=cp)
        if not math.isfinite(L):
            return _nan_result(n_iter)
        zeta = z_m / L

        ratio = sigma_T_over_Tstar(zeta)
        if not math.isfinite(ratio) or ratio == 0.0:
            return _nan_result(n_iter)

        abs_Tstar = sigma_T / ratio
        H_new = sign * rho * cp * ustar * abs_Tstar

        if abs(H_new - H) < tol:
            H = H_new
            converged = True
            break
        H = H_new

    # Final diagnostics consistent with the returned H. By definition
    # T* = -H / (rho cp ustar), so T* carries the opposite sign to H.
    L = obukhov_length(ustar, T_K, H, rho=rho, cp=cp)
    zeta = z_m / L if math.isfinite(L) else float("nan")
    Tstar = -sign * abs_Tstar
    return FVSResult(H=float(H), Tstar=float(Tstar), zeta=float(zeta),
                     n_iter=n_iter, converged=converged)
