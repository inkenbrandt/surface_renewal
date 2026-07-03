# Quick start

Prefer a runnable, end-to-end walkthrough? The
{doc}`tutorial notebooks <tutorials/01_quickstart>` cover the same ground
(and more) as executable Jupyter notebooks with embedded plots.

## Input data

The pipeline consumes high-frequency (10–20 Hz) scalar and wind data as a
`pandas.DataFrame` or a path to a CSV/Parquet file. Required columns:

| Column | Description | Units |
|--------|-------------|-------|
| `T` | Air (sonic or thermocouple) temperature | K or °C |
| `u`, `v`, `w` | Wind components | m s⁻¹ |
| `Rn` *(optional)* | Net radiation — enables the LE residual and the `daytime_only` screen | W m⁻² |
| `G` *(optional)* | Ground heat flux — enables the LE residual | W m⁻² |

The frame should carry a `DatetimeIndex`; if it does not, pass the name of the
timestamp column via `time_col`, or a synthetic index is built from the
sampling frequency. File loading goes through
{func}`surface_renewal.io.read_highfreq`, which also normalises common column
aliases (e.g. `Ts`, `Ux`, `Uy`, `Uz`).

## Running the pipeline

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

out = run_surface_renewal("my_highfreq_data.parquet", cfg=cfg)
print(out.head())
```

Internally each block passes through:

1. **Despiking** — Hampel (default) or Gaussian detector on `T`, `u`, `v`, `w`
   ({func}`surface_renewal.preprocess.despike.despike_dataframe`), plus a
   non-destructive physical-range QC screen.
2. **Coordinate rotation** — planar fit (default), double rotation, or none
   ({mod}`surface_renewal.preprocess.rotation`); the rotated components feed
   the friction-velocity estimate.
3. **Block diagnostics & stability screening** — u*, σ_T, S₃(τ*), and the
   configured thresholds decide the boolean `passed` column
   ({mod}`surface_renewal.preprocess.stability`).
4. **Flux estimation** — the configured SR method produces the uncalibrated
   sensible heat flux `H_uncal` for the block.

## Output columns

`run_surface_renewal` returns one row per block (indexed by block end time):

| Column | Meaning |
|--------|---------|
| `H_uncal` | Uncalibrated sensible heat flux (W m⁻²) |
| `H_cal` | Calibrated H — only when `alpha` is passed |
| `LE_resid` / `LE_cal` | Residual latent heat `Rn − G − H` (uncal./cal.) |
| `passed` | Block passed the stability screens |
| `ustar` | Friction velocity u* (m s⁻¹) from rotated covariances |
| `U_mean` | Block-mean horizontal wind speed (m s⁻¹) |
| `tau_star`, `dt_opt` | Ramp period / optimal structure-function lag (s) |
| `n_ramps` | Ramp count (wavelet method only) |
| `zeta` | Stability parameter z/L (height-dependent methods) |
| `alpha_sr` | Castellví analytic weighting factor (Castellví only) |
| `S3_tau` | Third-order structure function at τ* |
| `stdT` | Block temperature standard deviation (K) |
| `rho`, `cp` | Air density and specific heat used |
| `frac_qc_flagged` | Fraction of records flagged by the QC range screen |
| `CT2`, `CT2_r2` | Temperature structure parameter and fit quality |
| `flux_method_used` | `"primary"` or `"free_convection"` per block |

## Calibrating against eddy covariance

`snyder`, `chen97`, and `wavelet` return an **uncalibrated** magnitude that
must be scaled by a fitted block-scale factor `alpha` to match an
eddy-covariance (EC) reference:

```python
# ec_H is a block-indexed reference series (e.g. EC sensible heat).
cal = Calibration.from_reference(out["H_uncal"], ec_H)   # fits alpha (beta=0)
out = run_surface_renewal("my_highfreq_data.parquet", cfg=cfg, alpha=cal)
print(out[["H_uncal", "H_cal"]].head())
```

You can also pass a bare float (`alpha=1.15`). When `H_cal` is present and
both `Rn` and `G` are available, a calibrated residual
`LE_cal = Rn - G - H_cal` is added.

## Comparing methods

{func}`surface_renewal.compare_methods` preprocesses the data once and runs
every method on the identical cleaned series;
{func}`surface_renewal.method_agreement` summarises pairwise agreement:

```python
from surface_renewal import compare_methods, method_agreement

wide = compare_methods("my_highfreq_data.parquet", cfg=cfg)
print(method_agreement(wide))
```

The same comparison is available from the command line via
`python -m surface_renewal.compute ... --compare` (see {doc}`cli`).
