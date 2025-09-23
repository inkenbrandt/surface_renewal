import pandas as pd
import pytest

from surface_renewal.pipeline import PipelineConfig, run_surface_renewal, _ensure_df
from tests.helpers import generate_test_data

@pytest.fixture
def sample_data():
    """Return a sample DataFrame for testing."""
    return generate_test_data(fs=10, n_rows=20000, realistic=True)

def test_run_surface_renewal_snyder(sample_data):
    """Test the pipeline with the Snyder method."""
    config = PipelineConfig(fs=10, method="snyder", rotation="double")
    result = run_surface_renewal(sample_data, cfg=config)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "H_uncal" in result.columns
    assert "passed" in result.columns
    assert result["passed"].dtype == "bool"

def test_run_surface_renewal_chen97(sample_data):
    """Test the pipeline with the Chen97 method."""
    config = PipelineConfig(fs=10, method="chen97", rotation="double")
    result = run_surface_renewal(sample_data, cfg=config)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "H_uncal" in result.columns
    assert "passed" in result.columns
    assert result["passed"].dtype == "bool"

def test_run_surface_renewal_empty_data():
    """Test the pipeline with an empty DataFrame."""
    config = PipelineConfig(fs=10)
    empty_df = pd.DataFrame(columns=["T", "u", "v", "w"])
    result = run_surface_renewal(empty_df, cfg=config)

    assert isinstance(result, pd.DataFrame)
    assert result.empty

def test_ensure_df_no_datetimeindex(sample_data):
    """Test that _ensure_df creates a DatetimeIndex."""
    df_no_index = sample_data.reset_index(drop=True)

    # without a time_col, a synthetic index should be created
    df_synthetic_index = _ensure_df(df_no_index, fs=10)
    assert isinstance(df_synthetic_index.index, pd.DatetimeIndex)
    assert df_synthetic_index.index[0] == pd.to_datetime("1970-01-01 00:00:00")

    # with a time_col, it should be used for the index
    df_with_time = sample_data.reset_index().rename(columns={"index": "time"})
    df_with_index = _ensure_df(df_with_time, fs=10, time_col="time")
    assert isinstance(df_with_index.index, pd.DatetimeIndex)
