# src/surface_renewal/structure.py
"""Temperature structure functions for surface-renewal analysis.

This module provides the two primitives shared by the SR flux methods
(:mod:`surface_renewal.methods.snyder`, :mod:`surface_renewal.methods.chen97`)
and the stability screen (:mod:`surface_renewal.preprocess.stability`):

* :func:`structure_functions` -- the n-th order temperature structure function
  evaluated over a grid of integer lags, and
* :func:`pick_optimal_lag` -- Chen's ``r_m`` lag-selection criterion.

Physics
-------
Following Van Atta (1977), Snyder et al. (1996), and the SR formulation used
in AMT (2018), the n-th order temperature structure function at integer lag
``j`` (in samples) is the mean of the n-th power of the forward increment:

.. math::

    S_n(j) = \\frac{1}{N - j} \\sum_{i=j}^{N-1} \\bigl(T_i - T_{i-j}\\bigr)^n

Odd orders (notably :math:`S_3` and :math:`S_5`) carry the sign asymmetry of
the ramp structures that SR exploits; even orders (:math:`S_2`) measure
variance at the lag scale.

References
----------
Van Atta, C. W. (1977). Effect of coherent structures on structure functions
    of temperature in the atmospheric boundary layer. *Archives of Mechanics*,
    29, 161-171.
Snyder, R. L., Spano, D., & Paw U, K. T. (1996). Surface renewal analysis for
    sensible and latent heat flux density. *Boundary-Layer Meteorology*, 77,
    249-266.
Chen, W., Novak, M. D., Black, T. A., & Lee, X. (1997). Coherent eddies and
    temperature structure functions for three contrasting surfaces.
    *Boundary-Layer Meteorology*, 84, 99-124.
"""
from __future__ import annotations

import numpy as np

__all__ = ["structure_functions", "pick_optimal_lag"]


def structure_functions(
    T: np.ndarray,
    lags: np.ndarray,
    orders: list[int] = [2, 3, 5],
) -> dict[int, np.ndarray]:
    r"""Compute temperature structure functions over a grid of lags.

    For each requested order ``n`` and each integer lag ``j`` in ``lags``,
    evaluates the NaN-safe forward-increment structure function

    .. math::

        S_n(j) = \frac{1}{N_j} \sum_{i=j}^{N-1} \bigl(T_i - T_{i-j}\bigr)^n,

    where :math:`N_j` is the number of valid (non-NaN) increment pairs at lag
    ``j``.  Pairs in which either ``T[i]`` or ``T[i - j]`` is NaN are ignored,
    and the sum is divided by the count of valid pairs (not ``N - j``).

    Parameters
    ----------
    T : np.ndarray
        1-D high-frequency temperature series (K or degC). May contain NaNs.
    lags : np.ndarray
        Integer lags in **samples** at which to evaluate the structure
        functions. Values should be ``>= 1``; a lag of ``0`` yields ``S_n = 0``.
    orders : list[int], optional
        Structure-function orders to compute. Default ``[2, 3, 5]``.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping ``{order: array}`` where each array has the same length as
        ``lags`` and is aligned 1:1 with it, i.e. ``result[n][k]`` is
        :math:`S_n` evaluated at ``lags[k]``. A lag with no valid pairs yields
        ``NaN`` for every order.

    Notes
    -----
    Callers use this as::

        S = structure_functions(T, lags, orders=[2, 3, 5])
        S[3][k]   # == S_3 at lags[k]

    Examples
    --------
    >>> import numpy as np
    >>> T = np.arange(10.0)            # constant slope -> increment == j
    >>> S = structure_functions(T, np.array([1, 2]), orders=[2, 3])
    >>> S[2]
    array([1., 4.])
    >>> S[3]
    array([1., 8.])
    """
    T = np.asarray(T, dtype=float)
    if T.ndim != 1:
        raise ValueError(f"T must be 1-D, got shape {T.shape}")

    lags = np.asarray(lags)
    if lags.ndim != 1:
        raise ValueError(f"lags must be 1-D, got shape {lags.shape}")

    n = T.size
    out: dict[int, np.ndarray] = {
        int(order): np.full(lags.shape, np.nan, dtype=float) for order in orders
    }

    for k, j in enumerate(np.asarray(lags, dtype=int)):
        if j < 0:
            raise ValueError(f"lags must be non-negative, got {j}")
        if j == 0:
            # Zero lag: every increment is identically zero.
            for order in out:
                out[order][k] = 0.0
            continue
        if j >= n:
            # No pairs available at this lag; leave as NaN.
            continue

        # Forward increments: diff[i - j] = T[i] - T[i - j] for i in [j, N).
        diff = T[j:] - T[:-j]
        valid = np.isfinite(diff)
        count = int(valid.sum())
        if count == 0:
            continue

        dvalid = diff[valid]
        for order in out:
            out[order][k] = np.sum(dvalid ** order) / count

    return out


def pick_optimal_lag(S3: np.ndarray, lags: np.ndarray) -> int:
    r"""Select the optimal lag via Chen's ``r_m`` criterion.

    Chooses the lag ``r`` (in **samples**) that maximizes
    :math:`\lvert S_3(r) \rvert / r`. Because the ramp inverse time scale in the
    SR model scales as :math:`(-S_3 / r)^{1/3}`, maximizing :math:`|S_3| / r`
    is equivalent to maximizing :math:`-(S_3 / r)^{1/3}`; both pick the lag at
    which the third-order structure function grows fastest relative to the lag
    separation (Chen et al., 1997).

    Parameters
    ----------
    S3 : np.ndarray
        Third-order structure function values, aligned 1:1 with ``lags``
        (i.e. ``S3[k]`` corresponds to ``lags[k]``). May contain NaNs.
    lags : np.ndarray
        Candidate lags in **samples**, the same length as ``S3``.

    Returns
    -------
    int
        The element of ``lags`` that maximizes ``|S3| / lag``. The returned
        value is guaranteed to be a member of ``lags`` so that callers can do
        ``np.where(lags == k_opt)[0][0]`` safely.

    Raises
    ------
    ValueError
        If ``lags`` is empty, if ``S3`` and ``lags`` differ in length, or if no
        finite, positive-lag candidate exists (all-NaN ratios).

    Notes
    -----
    Lags ``<= 0`` are excluded from the search (guarding the ``r = 0``
    division) but, like NaN entries, do not by themselves cause an error as
    long as at least one valid candidate remains. Callers in this package
    (``snyder``, ``chen97``, ``stability``) index back into ``lags`` with the
    returned value, so a member of ``lags`` is always returned rather than an
    out-of-grid sentinel.
    """
    S3 = np.asarray(S3, dtype=float)
    lags = np.asarray(lags)

    if lags.size == 0:
        raise ValueError("pick_optimal_lag: `lags` is empty")
    if S3.shape != lags.shape:
        raise ValueError(
            f"pick_optimal_lag: S3 and lags must align; "
            f"got {S3.shape} vs {lags.shape}"
        )

    lags_f = lags.astype(float)

    # |S3| / r, with r <= 0 and non-finite ratios masked out of the argmax.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.abs(S3) / lags_f
    ratio = np.where(lags_f > 0, ratio, np.nan)

    if not np.any(np.isfinite(ratio)):
        raise ValueError(
            "pick_optimal_lag: no finite |S3|/lag candidate "
            "(all-NaN S3 or no positive lag)"
        )

    k = int(np.nanargmax(ratio))
    return int(lags[k])
