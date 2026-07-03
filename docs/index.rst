surface_renewal
===============

Surface renewal (SR) and micrometeorological sensible-heat flux (H) analysis
from 10–20 Hz temperature and wind measurements.

The package bundles **five** flux estimators — Snyder (1996), Chen (1997),
flux–variance similarity (FVS), Castellví (2004), and a wavelet ramp detector
(Collineau & Brunet 1993) — behind a single preprocessing and block-averaging
pipeline, plus a free-convection fallback for low-wind convective periods.
Latent heat (LE) is available as an energy-balance residual when net radiation
(``Rn``) and ground heat flux (``G``) are provided.

.. code-block:: python

   from surface_renewal import run_surface_renewal, PipelineConfig

   cfg = PipelineConfig(fs=20.0, block="30min", method="snyder")
   out = run_surface_renewal("my_highfreq_data.parquet", cfg=cfg)
   print(out[["H_uncal", "tau_star", "passed"]].head())

Highlights
----------

- **One pipeline, five methods** — despiking, coordinate rotation, stability
  screening, and block averaging are shared; switch the flux estimator with a
  single ``method=`` argument.
- **Calibration-free options** — ``fvs`` and ``castellvi`` close the flux
  through Monin–Obukhov similarity theory, so no eddy-covariance tower is
  needed; ``snyder``, ``chen97``, and ``wavelet`` accept a fitted block-scale
  ``alpha`` when a reference is available.
- **Free-convection fallback** — strongly unstable, low-wind blocks can fall
  back to a :math:`\sigma_T`-only estimate where :math:`u_*`-based methods
  degrade.
- **Method comparison tools** — run all methods on identical preprocessed data
  and quantify pairwise agreement.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   quickstart
   methods
   cli

.. toctree::
   :maxdepth: 2
   :caption: Background

   theory

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
