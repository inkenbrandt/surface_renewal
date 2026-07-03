# Choosing a method

Five surface-renewal (SR) flux estimators share one preprocessing and
block-averaging pipeline; select one with `PipelineConfig(method=...)`. The
governing equations for each are collected in {doc}`theory`.

| Method | Reference | Needs u\*? | Needs z_m? | Needs EC calibration (alpha)? | Stability range |
|--------|-----------|:----------:|:----------:|:-----------------------------:|-----------------|
| `snyder` | Snyder et al. (1996); Van Atta (1977) | no | no | **yes** | Unstable / daytime warming ramps (magnitude only) |
| `chen97` | Chen et al. (1997) | **yes** | no | **yes** | All stabilities (sign from S₃); degrades as u\* → 0 |
| `fvs` | Tillman (1972); Katul et al. (1995) | **yes** | **yes** | no | Unstable (near-neutral/stable floor is weak) |
| `castellvi` | Castellví (2004); Castellví & Snyder (2009) | **yes** | **yes** | no | Unstable / daytime warming ramps |
| `wavelet` | Collineau & Brunet (1993) | no | no | **yes** | Any / non-stationary conditions |
| free-convection fallback | Tillman (1972) | no | **yes** | no | Strongly unstable, low-wind only (−z/L ≫ 1) |

**Needs EC calibration (alpha):** `snyder`, `chen97`, and `wavelet` return an
uncalibrated $H = \rho c_p (A/\tau)$-style magnitude that must be scaled by a
fitted `alpha` to match an eddy-covariance (EC) reference. `fvs` and
`castellvi` are **calibration-free** — FVS closes H through Monin–Obukhov
similarity theory (MOST), and Castellví derives the SR weighting factor
analytically — so no EC tower is required.

## Decision guide

- **No EC tower available?** Use `castellvi` or `fvs` — both close the flux
  through MOST without any site calibration (`castellvi` needs `z_m` and u\*;
  `fvs` needs `z_m`, u\*, and gives magnitude with the sign taken from
  `sign(S3(tau*))`).
- **EC tower available for calibration?** Use `snyder` (no u\* needed) or
  `chen97` (uses u\*), then fit `alpha` with
  {meth}`Calibration.from_reference <surface_renewal.preprocess.calibration.Calibration.from_reference>`
  and apply it.
- **Low-wind convective site** where u\* → 0 makes the u\*-based methods
  unreliable: set `free_convection_fallback=True` (requires `z_m`) so strongly
  unstable, low-wind blocks fall back to the free-convection estimate.
- **Non-stationary conditions** (transitions, intermittent turbulence) where
  the structure-function ramp recovery is unstable: use `wavelet`, which
  locates the dominant ramp scale from the wavelet variance and counts
  renewals from zero-crossings.

## The free-convection fallback

The fallback is not a standalone `method=`; it is enabled with
`PipelineConfig(free_convection_fallback=True)` and substitutes the
$\sigma_T$-only estimate
({func}`surface_renewal.methods.fvs.estimate_H_free_convection`) for a
u\*-based primary method (`chen97`, `fvs`, `castellvi`) on a block only when
that block is:

- **low-wind** — `ustar < fc_ustar_max` (default `0.1` m s⁻¹), and
- **strongly unstable** — `zeta = z_m / L < fc_zeta_max` (default `−0.5`),
  with `L` computed from the primary H, and
- the primary H was itself **positive** (free convection implies upward flux;
  the fallback is never applied at night or to downward fluxes).

The per-block choice is recorded in the `flux_method_used` output column
(`"primary"` or `"free_convection"`). The fallback requires `z_m`.

## About `z_m` (measurement height)

`z_m` is the sensor height **above the zero-plane displacement**, not above
the ground:

```text
z_m = z_sensor - d,   with   d ≈ 0.66 * canopy_height
```

where `d` is the zero-plane displacement height and `canopy_height` is the
mean height of the vegetation/roughness elements. For example, a sonic at
`z_sensor = 3.0 m` over a `1.0 m` canopy gives `d ≈ 0.66 m` and
`z_m ≈ 2.34 m`. The height-dependent methods (`fvs`, `castellvi`) and the
free-convection fallback raise a `ValueError` if `z_m` is left as `None`.
