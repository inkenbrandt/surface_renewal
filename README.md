# srflux

Surface renewal (SR) flux analysis from 10–20 Hz temperature and wind. Implements
Snyder96 and Chen97 formulations for sensible heat (H); computes latent heat (LE)
as energy‑balance residual if Rn and G are provided.

## Quick start

```python
from snyder96 import compute_fluxes, compute_block_diagnostics, stability_ok

res = compute_fluxes(
    path="my_highfreq_data.parquet",
    time_col=None,           # if the file already has a DatetimeIndex
    model="chen97",          # or "snyder96"
    block="30min",
    despike=True,
    rotation="planar_fit",   # none | double | planar_fit (for Chen97 u*)
    # Stability screens (tune as needed):
    min_ustar=0.05,
    min_rel_S3=1e-3,
    min_stdT=0.02,
    daytime_only=False,
    alpha=None,              # auto‑calibrate if H_ref provided; else 1.0
)
print(res.head())
```

Input columns required: `T` (K), `u`,`v`,`w` (m s‑1). Optional: `Rn`, `G` (W m‑2).

## Testing
Run the tests with `pytest`:

```bash
pytest -q
```

---