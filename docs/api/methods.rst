Flux estimators
===============

Each method module exposes an ``estimate_H_*`` (or ``detect_ramps_*``)
function returning a ``NamedTuple`` result. The theory behind each estimator
is documented in :doc:`../theory`.

Snyder (1996) cubic-ramp method
-------------------------------

.. automodule:: surface_renewal.methods.snyder

Chen (1997) method
------------------

.. automodule:: surface_renewal.methods.chen97

Flux–variance similarity (FVS) & free convection
------------------------------------------------

.. automodule:: surface_renewal.methods.fvs

Castellví (2004) calibration-free method
----------------------------------------

.. automodule:: surface_renewal.methods.castellvi

Wavelet ramp detection (Collineau & Brunet 1993)
------------------------------------------------

.. automodule:: surface_renewal.methods.wavelet

Method comparison & ramp analysis
---------------------------------

.. automodule:: surface_renewal.methods.analysis
