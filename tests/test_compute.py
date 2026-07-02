import os
import pytest
from unittest.mock import patch
import numpy as np
import pandas as pd

from surface_renewal.compute import main
from surface_renewal.structure import structure_functions
from tests.helpers import generate_test_data


def _structure_functions_loop(T, lags, orders=(2, 3, 5)):
    """Reference per-lag loop implementation for cross-checking."""
    T = np.asarray(T, float)
    lags = np.asarray(list(lags), int)
    out = {p: np.full(lags.size, np.nan, float) for p in orders}
    n = T.size
    for i, k in enumerate(lags):
        if k <= 0 or k >= n:
            continue
        d = T[k:] - T[:-k]
        for p in orders:
            out[p][i] = (
                np.nanmean(np.abs(d) ** p) if p % 2 == 0
                else np.nanmean(d ** p)
            )
    return out


def test_structure_functions_matches_loop():
    """Vectorized structure_functions matches the naive loop on a ramp."""
    t = np.arange(300.0)
    T = np.sin(t / 5.0) + 0.05 * t  # synthetic ramp with structure
    lags = [1, 2, 3, 5, 8, 13, 21, 299, 300]  # include boundary/invalid lags
    orders = (2, 3, 5)

    fast = structure_functions(T, lags, orders=orders)
    ref = _structure_functions_loop(T, lags, orders=orders)

    for p in orders:
        assert np.allclose(fast[p], ref[p], equal_nan=True)

@pytest.fixture
def sample_data_file(tmp_path):
    """Create a sample data file for testing."""
    fpath = tmp_path / "sample_data.parquet"
    data = generate_test_data(fs=10, n_rows=20000, realistic=True)
    data.to_parquet(fpath)
    return str(fpath)

def test_compute_cli_snyder(sample_data_file):
    """Test the CLI with the Snyder method."""
    args = [
        "surface_renewal.compute",
        sample_data_file,
        "--fs", "10",
        "--method", "snyder",
        "--rotation", "double",
    ]
    with patch("sys.argv", args):
        assert main() == 0

def test_compute_cli_chen97(sample_data_file):
    """Test the CLI with the Chen97 method."""
    args = [
        "surface_renewal.compute",
        sample_data_file,
        "--fs", "10",
        "--method", "chen97",
        "--rotation", "double",
    ]
    with patch("sys.argv", args):
        assert main() == 0

def test_compute_cli_output_file(sample_data_file, tmp_path):
    """Test the CLI with an output file."""
    output_fpath = tmp_path / "output.csv"
    args = [
        "surface_renewal.compute",
        sample_data_file,
        "--fs", "10",
        "--method", "snyder",
        "--rotation", "double",
        "--out", str(output_fpath),
    ]
    with patch("sys.argv", args):
        assert main() == 0

    assert output_fpath.exists()
    # check that the output file is a valid csv
    df = pd.read_csv(output_fpath)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
