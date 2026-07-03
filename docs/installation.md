# Installation

`surface_renewal` requires Python 3.9 or newer.

## From source

The package is currently distributed from its Git repository:

```bash
git clone https://github.com/inkenbrandt/surface_renewal.git
cd surface_renewal
pip install .
```

For development (editable install plus test dependencies):

```bash
pip install -e .
pip install pytest
pytest -q
```

If you use [uv](https://docs.astral.sh/uv/), the repository ships a
`uv.lock`, so `uv sync` will create a matching environment.

## Dependencies

Installed automatically by `pip`:

| Package | Minimum version | Used for |
|---------|-----------------|----------|
| numpy | 1.22 | Array math throughout |
| scipy | 1.10 | Signal/statistics helpers |
| pandas | 2.0 | Time indexing, block averaging, I/O |
| xarray | 2023.1.0 | Labeled outputs |
| numba | 0.58 | Accelerated kernels |
| pyarrow | 12.0 | Parquet reading/writing |
| matplotlib | 3.8 | Optional plotting |

## Verifying the install

```python
import surface_renewal
print(surface_renewal.__version__)
```

You can also run the built-in test suite from a source checkout with
`pytest -q`.
