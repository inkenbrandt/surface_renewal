# Command-line interface

The pipeline can be run without writing any Python via the module CLI:

```bash
python -m surface_renewal.compute my_highfreq_data.parquet --fs 20 --method snyder --out fluxes.parquet
```

The positional argument is a path to a CSV or Parquet file with at least the
`T`, `u`, `v`, `w` columns (see {doc}`quickstart` for the input schema). If
`--out` is omitted, a preview of the last blocks is printed to stdout;
otherwise results are written with
{func}`surface_renewal.io.write_flux_timeseries`.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--fs` | *(required)* | Sampling frequency (Hz) |
| `--block` | `30min` | Block-averaging period (pandas offset alias) |
| `--method` | `snyder` | `snyder`, `chen97`, `fvs`, `castellvi`, or `wavelet` |
| `--rotation` | `planar_fit` | `planar_fit`, `double`, or `none` |
| `--despike` | `hampel` | Spike detector: `hampel` or `gaussian` |
| `--hampel-window` | `11` | Hampel window length (samples) |
| `--hampel-sigmas` | `3.0` | Hampel MAD threshold multiplier |
| `--gauss-nw` | `201` | Gaussian despiker window (samples) |
| `--gauss-sig` | `4.0` | Gaussian despiker sigma threshold |
| `--gauss-buffer` | `3` | Buffer (samples) around flagged spikes |
| `--interp` | *(none)* | Interpolate spike gaps: `linear`, `nearest`, `cubic` |
| `--interp-max-gap` | *(none)* | Max consecutive NaNs (samples) to interpolate |
| `--min-ustar` | `0.05` | Minimum u* (m s⁻¹) to accept a block |
| `--min-relS3` | `1e-3` | Minimum \|S3(τ*)\| / std(T)³ to accept a block |
| `--min-stdT` | `0.02` | Minimum std(T) (K) to accept a block |
| `--daytime-only` | off | Require Rn > 0 (when Rn is provided) |
| `--z-m` | *(none)* | Height above zero-plane displacement (m); required by `fvs`/`castellvi` |
| `--free-convection-fallback` | off | Enable the free-convection fallback (requires `--z-m`) |
| `--fc-ustar-max` | `0.1` | Upper u* bound (m s⁻¹) for the fallback |
| `--fc-zeta-max` | `-0.5` | Fallback applies when block ζ < this value |
| `--time-col` | *(none)* | Timestamp column name if the file lacks a datetime index |
| `--out` | *(none)* | Output Parquet/CSV path |
| `--compare` | off | Run **all** methods and print the pairwise agreement table |

## Comparison mode

`--compare` runs every SR method on the same preprocessed data (ignoring
`--method`) and prints per-block results plus the pairwise method-agreement
table from {func}`surface_renewal.method_agreement`. Height-dependent methods
need `--z-m`:

```bash
python -m surface_renewal.compute my_highfreq_data.parquet --fs 20 --z-m 2.34 --compare
```
