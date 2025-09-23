import os
import pytest
from unittest.mock import patch
import pandas as pd

from surface_renewal.compute import main
from tests.helpers import generate_test_data

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
