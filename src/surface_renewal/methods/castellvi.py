# src/surface_renewal/methods/castellvi.py
r"""Castellví (2004) calibration-free surface-renewal sensible heat flux.

Classical surface-renewal (SR) analysis estimates the sensible heat flux from
the ramp amplitude ``A`` and mean ramp period ``τ`` recovered from a
high-frequency temperature series as

.. math::

    H = \rho\, c_p\, \alpha\, \frac{A}{\tau},

where ``α`` is an empirical weighting factor that historically had to be fitted
against an eddy-covariance (EC) reference. Castellví (2004) removed the need for
that calibration by deriving ``α`` analytically: he combined SR analysis with
Monin–Obukhov similarity theory (MOST) through the dissipation rate of
temperature variance. For measurements in the inertial sublayer (above the
roughness sublayer) the balance between production and dissipation of
temperature variance yields a closed form for ``α`` in terms of the friction
velocity :math:`u_*`, the measurement height :math:`z_m` (above the zero-plane
displacement), the ramp period :math:`\tau`, and the MOST temperature-gradient
stability function :math:`\phi_h(\zeta)`:

.. math::

    \alpha = \sqrt{\frac{\pi\,\kappa\,z_m}{4\,\tau\,u_*\,\phi_h(\zeta)}}.

The argument of the square root is dimensionless — :math:`\kappa z_m` has units
of metres, and :math:`\tau u_*` metres as well (``s · m s⁻¹``), while
:math:`\pi/4` and :math:`\phi_h` are dimensionless — so ``α`` is dimensionless,
consistent with the SR convention used throughout this package (the same
convention under which :func:`surface_renewal.methods.snyder.estimate_H_snyder`
forms ``H = ρ c_p (A/τ)`` without an explicit height factor).

Because ``α`` depends on :math:`\zeta = z_m / L` and the Obukhov length ``L``
itself depends on ``H``, the estimate is solved iteratively: start from neutral
(:math:`\zeta = 0`, :math:`\phi_h = 1`), form ``H``, update ``L`` and
:math:`\zeta`, and repeat until ``H`` converges. The ramp sign is carried by
``A`` (a warming ramp gives ``A > 0`` and ``H > 0``), so ``H`` is signed
naturally without a separate sign hint.

The Obukhov length and the stability functions :math:`\phi_h` / :math:`f_{C_T^2}`
are imported from :mod:`surface_renewal.most` so that every method shares one
definition of the MOST relations, and the ``(A, τ)`` recovery is imported from
:mod:`surface_renewal.methods.snyder` so Castellví and Snyder see identical ramp
characteristics.

Verification note
-----------------
The inertial-sublayer form implemented here — the :math:`\sqrt{\pi\kappa z_m /
(4\tau u_* \phi_h)}` weighting factor — was checked for dimensional consistency
and against the ingredients of Castellví's derivation as reproduced in the
secondary literature (the dissipation-method combination of SR with MOST
involving :math:`u_*`, :math:`\phi_h`, :math:`\kappa`, :math:`z_m`, and
:math:`\tau`; e.g. Castellví & Snyder 2009 and Mengistu & Savage 2010). The
primary WRR (2004) PDF was not machine-readable during implementation, so the
exact equation numbers below should be confirmed against the published paper; if
the paper's prefactor or the placement/exponent of :math:`\phi_h` differs from
the form above, prefer the paper and update this module accordingly.

References
----------
Castellví, F. (2004). Combining surface renewal analysis and similarity theory:
    A new approach for estimating sensible heat flux. *Water Resources
    Research*, 40, W05201. doi:10.1029/2003WR002677. (Inertial-sublayer
    weighting factor, eqs. ~10–12.)
Castellví, F., & Snyder, R. L. (2009). Combining the dissipation method and
    surface renewal analysis to estimate scalar fluxes from the low frequency
    of scalar high-frequency measurements. *Journal of Hydrology*, 373(3–4),
    142–151. doi:10.1016/j.jhydrol.2009.04.020.
Mengistu, M. G., & Savage, M. J. (2010). Surface renewal method for estimating
    sensible heat flux. *Water SA*, 36(1), 9–18.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from ..most import KAPPA, obukhov_length, phi_h, f_CT2
from .snyder import recover_ramp


class CastellviResult(NamedTuple):
    """Result of a Castellví (2004) calibration-free SR sensible-heat estimate.

    Attributes
    ----------
    H : float
        Sensible heat flux (W m⁻²). Signed via the ramp amplitude ``A``.
    alpha : float
        Analytic SR weighting factor :math:`\\alpha` at convergence
        (dimensionless).
    A : float
        Ramp amplitude (K) recovered by the Van Atta cubic model.
    tau : float
        Mean ramp period :math:`\\tau` (s).
    zeta : float
        Stability parameter :math:`\\zeta = z_m / L` at convergence.
    n_iter : int
        Number of stability iterations performed.
    converged : bool
        Whether the iteration met the convergence tolerance.
    """
    H: float
    alpha: float
    A: float
    tau: float
    zeta: float
    n_iter: int
    converged: bool


def _nan_result(A: float = float("nan"), tau: float = float("nan"),
                n_iter: int = 0) -> CastellviResult:
    """Return an all-NaN, non-converged result (keeping any recovered A, τ)."""
    nan = float("nan")
    return CastellviResult(H=nan, alpha=nan, A=A, tau=tau, zeta=nan,
                           n_iter=n_iter, converged=False)


def alpha_castellvi(z_m: float, tau: float, ustar: float, zeta: float) -> float:
    r"""Analytic Castellví (2004) inertial-sublayer weighting factor.

    .. math::

        \alpha = \sqrt{\frac{\pi\,\kappa\,z_m}{4\,\tau\,u_*\,\phi_h(\zeta)}}.

    Parameters
    ----------
    z_m : float
        Measurement height above the zero-plane displacement (m).
    tau : float
        Mean ramp period :math:`\tau` (s).
    ustar : float
        Friction velocity :math:`u_*` (m s⁻¹).
    zeta : float
        Stability parameter :math:`\zeta = z_m / L`. At neutral (``zeta == 0``)
        :math:`\phi_h = 1`.

    Returns
    -------
    float
        The dimensionless weighting factor, or NaN if any input is non-finite,
        if :math:`\tau u_* \phi_h \le 0`, or if the radicand is negative.
    """
    if not all(math.isfinite(x) for x in (z_m, tau, ustar, zeta)):
        return float("nan")
    ph = phi_h(zeta)
    denom = 4.0 * tau * ustar * ph
    if not math.isfinite(denom) or denom <= 0.0:
        return float("nan")
    radicand = (math.pi * KAPPA * z_m) / denom
    if radicand < 0.0:
        return float("nan")
    return float(math.sqrt(radicand))


def estimate_H_castellvi(
    T: np.ndarray,
    *,
    hz: float,
    ustar: float,
    T_K: float,
    z_m: float,
    rho: float = 1.2,
    cp: float = 1005.0,
    max_iter: int = 20,
    tol: float = 1e-3,
) -> CastellviResult:
    r"""Estimate sensible heat flux by the calibration-free Castellví (2004) SR method.

    Algorithm
    ---------
    a. Recover the ramp amplitude ``A`` and period ``τ`` with the Van Atta cubic
       machinery shared with Snyder (:func:`recover_ramp`).
    b. Iterate the stability loop, starting from neutral (:math:`\zeta = 0`,
       :math:`\phi_h = 1`):

       .. math::

           \alpha = \sqrt{\frac{\pi\,\kappa\,z_m}{4\,\tau\,u_*\,\phi_h(\zeta)}},
           \quad
           H = \rho\, c_p\, \alpha\, \frac{A}{\tau},
           \quad
           L = -\frac{\rho c_p T_K u_*^3}{\kappa g H},
           \quad
           \zeta = \frac{z_m}{L},

       until :math:`|H_\mathrm{new} - H| < \mathrm{tol}`.
    c. The ramp amplitude ``A`` carries the ramp sign, so ``H`` is signed
       naturally (:math:`H = \rho c_p \alpha (A/\tau)` with
       :math:`\alpha, \tau > 0`, hence ``sign(H) == sign(A)``).

    .. note::

       The shared :func:`recover_ramp` selects the *maximum* real root of the
       Van Atta cubic, which yields a valid positive ``τ`` only for **warming**
       ramps (:math:`S_3(\tau^*) < 0`, i.e. daytime unstable, ``A > 0``,
       ``H > 0``) — exactly the regime for which Castellví's inertial-sublayer
       :math:`\phi_h`/MOST derivation is intended. Cooling ramps
       (:math:`S_3 > 0`) do not produce a valid ``τ`` under that root
       convention and return a NaN, non-converged result, mirroring the
       magnitude-only behaviour of :func:`estimate_H_snyder`.

    Parameters
    ----------
    T : array_like
        High-frequency temperature series over the block (K or °C).
    hz : float
        Sampling frequency (Hz).
    ustar : float
        Friction velocity :math:`u_*` (m s⁻¹).
    T_K : float
        Block-mean air temperature (K), used in the Obukhov length.
    z_m : float
        Measurement height above the zero-plane displacement (m).
    rho : float, default 1.2
        Air density (kg m⁻³).
    cp : float, default 1005.0
        Specific heat of air at constant pressure (J kg⁻¹ K⁻¹).
    max_iter : int, default 20
        Maximum number of stability iterations.
    tol : float, default 1e-3
        Convergence tolerance on successive ``H`` values (W m⁻²).

    Returns
    -------
    CastellviResult
        The estimated flux, weighting factor, ramp characteristics, stability
        parameter, iteration count, and convergence flag. A NaN-filled,
        non-converged result is returned when ``ustar``, ``T_K``, or ``z_m`` is
        non-finite, when ``ustar < 0.01`` (too weak for a reliable MOST
        inversion), or when the ramp recovery is degenerate.
    """
    # Degenerate-input guards, mirroring the other height-dependent methods.
    if not (math.isfinite(ustar) and math.isfinite(T_K) and math.isfinite(z_m)):
        return _nan_result()
    if ustar < 0.01 or z_m <= 0.0:
        return _nan_result()

    # (a) Ramp recovery shared with Snyder. A carries the ramp sign.
    A, tau, _dt_opt = recover_ramp(T, hz=hz)
    if not (np.isfinite(A) and np.isfinite(tau)) or tau <= 0.0:
        return _nan_result(A=A, tau=tau)

    # (b) Stability loop, starting from neutral (zeta = 0 -> phi_h = 1).
    zeta = 0.0
    alpha = alpha_castellvi(z_m, tau, ustar, zeta)
    if not math.isfinite(alpha):
        return _nan_result(A=A, tau=tau)
    H = rho * cp * alpha * (A / tau)

    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        L = obukhov_length(ustar, T_K, H, rho=rho, cp=cp)
        if not math.isfinite(L):
            return _nan_result(A=A, tau=tau, n_iter=n_iter)
        zeta = z_m / L

        alpha = alpha_castellvi(z_m, tau, ustar, zeta)
        if not math.isfinite(alpha):
            return _nan_result(A=A, tau=tau, n_iter=n_iter)

        H_new = rho * cp * alpha * (A / tau)
        if abs(H_new - H) < tol:
            H = H_new
            converged = True
            break
        H = H_new

    return CastellviResult(H=float(H), alpha=float(alpha), A=float(A),
                           tau=float(tau), zeta=float(zeta), n_iter=n_iter,
                           converged=converged)


def alpha_from_CT2(CT2: float, z_m: float, Tstar: float, zeta: float) -> float:
    r"""Independent cross-check on ``α`` via the temperature structure parameter.

    This is a **QA cross-check**, not part of the :func:`estimate_H_castellvi`
    flux calculation. It offers a second, independent route to the temperature
    scale that underpins the SR weighting factor, using the MOST scaling of the
    temperature structure parameter :math:`C_T^2` (Wyngaard et al. 1971):

    .. math::

        T_*^2 = \frac{C_T^2\, z_m^{2/3}}{f_{C_T^2}(\zeta)},

    where :math:`f_{C_T^2}` is :func:`surface_renewal.most.f_CT2`. The value
    returned here is the :math:`T_*` implied by ``CT2`` at stability ``zeta``;
    comparing it with the :math:`T_* = -H / (\rho c_p u_*)` implied by the
    primary Castellví estimate is a useful consistency check on ``α``. A large
    discrepancy flags a problem with the ramp recovery, the inertial-subrange
    ``CT2`` fit, or the stability estimate.

    Parameters
    ----------
    CT2 : float
        Temperature structure parameter :math:`C_T^2` (K² m⁻²ᐟ³), e.g. from
        :func:`surface_renewal.structure.estimate_CT2`.
    z_m : float
        Measurement height above the zero-plane displacement (m).
    Tstar : float
        The primary-route temperature scale :math:`T_*` (K) to compare against.
        Included so callers can request the signed cross-check magnitude with a
        matching sign; only its sign is used.
    zeta : float
        Stability parameter :math:`\zeta = z_m / L`.

    Returns
    -------
    float
        The :math:`T_*` magnitude implied by ``CT2`` (signed to match
        ``Tstar``), or NaN for non-finite input or a non-positive radicand.
    """
    if not all(math.isfinite(x) for x in (CT2, z_m, Tstar, zeta)):
        return float("nan")
    f = f_CT2(zeta)
    if not math.isfinite(f) or f <= 0.0 or CT2 < 0.0 or z_m <= 0.0:
        return float("nan")
    Tstar_sq = CT2 * z_m ** (2.0 / 3.0) / f
    if Tstar_sq < 0.0:
        return float("nan")
    mag = math.sqrt(Tstar_sq)
    sign = -1.0 if Tstar < 0.0 else 1.0
    return float(sign * mag)
