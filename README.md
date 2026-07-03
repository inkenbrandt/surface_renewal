# surface_renewal

Surface renewal (SR) and micrometeorological sensible-heat flux (H) analysis from
10–20 Hz temperature and wind. The package bundles **five** flux estimators —
Snyder (1996), Chen (1997), flux–variance similarity (FVS), Castellví (2004), and
a wavelet ramp detector (Collineau & Brunet 1993) — behind one preprocessing +
block-averaging pipeline, plus a free-convection fallback for low-wind convective
periods. Latent heat (LE) is available as an energy-balance residual when `Rn` and
`G` are provided.

The mathematics of each method (governing equations, iteration schemes, and the
full reference list) is documented in [docs/theory.md](docs/theory.md).

## Documentation

Full documentation (installation, quick start, method-selection guide, CLI
reference, theory, and API reference) lives in [docs/](docs/) and is built with
Sphinx for [Read the Docs](https://readthedocs.org/) (see
[.readthedocs.yaml](.readthedocs.yaml)). Three runnable Jupyter notebook
tutorials — a pipeline quickstart, a five-method comparison, and an
eddy-covariance calibration walkthrough — live in
[docs/tutorials/](docs/tutorials/); each is self-contained (it generates its
own synthetic data) and is rendered into the documentation with its outputs.
Build the docs locally with:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

## Quick start

```python
from surface_renewal import run_surface_renewal, PipelineConfig, Calibration

cfg = PipelineConfig(
    fs=20.0,                 # sampling frequency (Hz)
    block="30min",           # block-averaging period (pandas offset alias)
    method="snyder",         # snyder | chen97 | fvs | castellvi | wavelet
    rotation="planar_fit",   # none | double | planar_fit (used for u*)
    # Stability screens (tune as needed):
    stability_ustar=0.05,    # minimum u* (m s-1)
    stability_relS3=1e-3,    # minimum |S3(tau*)| / std(T)^3
    stability_stdT=0.02,     # minimum std(T) (K)
    daytime_only=False,
    z_m=None,                # measurement height above d (m); required by fvs/castellvi
)

# `data` is a high-frequency DataFrame (or a path to CSV/Parquet).
out = run_surface_renewal("my_highfreq_data.parquet", cfg=cfg, time_col=None)
print(out.head())
```

`run_surface_renewal` returns **uncalibrated** H (`H_uncal`) by design. To obtain a
calibrated H when an eddy-covariance (EC) reference is available, fit a single
block-scale factor `alpha` and pass it back:

```python
# ec_H is a block-indexed reference series (e.g. EC sensible heat).
cal = Calibration.from_reference(out["H_uncal"], ec_H)   # fits alpha (beta=0)
out = run_surface_renewal("my_highfreq_data.parquet", cfg=cfg, alpha=cal)
print(out[["H_uncal", "H_cal"]].head())
```

You can also pass a bare float (`alpha=1.15`). When `H_cal` is present and both
`Rn` and `G` are available, a calibrated residual `LE_cal = Rn - G - H_cal` is added.

Input columns required: `T` (K or °C), `u`, `v`, `w` (m s⁻¹). Optional: `Rn`, `G`
(W m⁻²) for the LE residual.

A back-compatible `compute(data, cfg=ComputeConfig(...))` wrapper and a CLI
(`python -m surface_renewal.compute ...`, including a `--compare` mode) are also
provided.

## Methods

| Method | Reference | Needs u\*? | Needs z\_m? | Needs EC calibration (alpha)? | Stability range |
|--------|-----------|:----------:|:-----------:|:-----------------------------:|-----------------|
| `snyder` | Snyder et al. (1996); Van Atta (1977) | no | no | **yes** | Unstable / daytime warming ramps (magnitude only) |
| `chen97` | Chen et al. (1997) | **yes** | no | **yes** | All stabilities (sign from S₃); degrades as u\* → 0 |
| `fvs` | Tillman (1972); Katul et al. (1995) | **yes** | **yes** | no | Unstable (near-neutral/stable floor is weak) |
| `castellvi` | Castellví (2004); Castellví & Snyder (2009) | **yes** | **yes** | no | Unstable / daytime warming ramps |
| `wavelet` | Collineau & Brunet (1993) | no | no | **yes** | Any / non-stationary conditions |
| free-convection fallback | Tillman (1972) | no | **yes** | no | Strongly unstable, low-wind only (−z/L ≫ 1) |

Notes:

- **Needs EC calibration (alpha):** `snyder`, `chen97`, and `wavelet` return an
  uncalibrated `H = ρ c_p (A/τ)`-style magnitude that must be scaled by a fitted
  `alpha` to match an EC reference. `fvs` and `castellvi` are **calibration-free**
  — FVS closes H through Monin–Obukhov similarity theory (MOST), and Castellví
  derives the SR weighting factor analytically — so no EC tower is required.
- The free-convection fallback is not a standalone `method=`; it is enabled with
  `free_convection_fallback=True` and substitutes for a u\*-based primary method
  (`chen97`, `fvs`, `castellvi`) on qualifying blocks. The choice made per block is
  recorded in the `flux_method_used` output column.

### Which method should I use?

- **No EC tower available?** Use `castellvi` or `fvs` — both close the flux through
  MOST without any site calibration (`castellvi` needs `z_m` and u\*; `fvs` needs
  `z_m`, u\*, and gives magnitude with the sign taken from `sign(S3(tau*))`).
- **EC tower available for calibration?** Use `snyder` (no u\* needed) or `chen97`
  (uses u\*), then fit `alpha` with
  `Calibration.from_reference(H_uncal, ec_H)` and apply it.
- **Low-wind convective site** where u\* → 0 makes the u\*-based methods unreliable:
  set `free_convection_fallback=True` (requires `z_m`) so strongly unstable,
  low-wind blocks fall back to the free-convection estimate.
- **Non-stationary conditions** (transitions, intermittent turbulence) where the
  structure-function ramp recovery is unstable: use `wavelet`, which locates the
  dominant ramp scale from the wavelet variance and counts renewals from
  zero-crossings.

### About `z_m` (measurement height)

`z_m` is the sensor height **above the zero-plane displacement**, not above the
ground:

```
z_m = z_sensor - d,   with   d ≈ 0.66 * canopy_height
```

where `d` is the zero-plane displacement height and `canopy_height` is the mean
height of the vegetation/roughness elements. For example, a sonic at
`z_sensor = 3.0 m` over a `1.0 m` canopy gives `d ≈ 0.66 m` and `z_m ≈ 2.34 m`.
The height-dependent methods (`fvs`, `castellvi`) and the free-convection fallback
raise a `ValueError` if `z_m` is left as `None`.

## Testing

Run the tests with `pytest`:

```bash
pytest -q
```
