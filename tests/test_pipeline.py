import numpy as np
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
    assert "U_mean" in result.columns
    assert (result["U_mean"] > 0).all()

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

def _chen97_segment(fs: int = 10) -> pd.DataFrame:
    """Build a synthetic 30-minute, DatetimeIndex-indexed segment for Chen97."""
    rng = np.random.default_rng(0)
    n_rows = fs * 60 * 30  # 30 minutes at `fs` Hz
    t = np.arange(n_rows) / fs

    # Sinusoidal ramp pattern for T: amplitude 0.3 K, period 60 s, plus noise.
    T = 298.15 + 0.3 * np.sin(2 * np.pi * t / 60.0) + rng.standard_normal(n_rows) * 0.02

    # White-noise wind components with the requested standard deviations.
    u = rng.standard_normal(n_rows) * 1.5
    v = rng.standard_normal(n_rows) * 0.3
    w = rng.standard_normal(n_rows) * 0.15

    start = pd.Timestamp("2023-01-01 00:00:00")
    index = start + pd.to_timedelta(t, unit="s")
    return pd.DataFrame({"T": T, "u": u, "v": v, "w": w}, index=index)


def test_chen97_pipeline_runs():
    """The Chen97 pipeline runs end-to-end on a synthetic ramp segment."""
    df = _chen97_segment(fs=10)

    config = PipelineConfig(fs=10, method="chen97", rotation="double", block="30min")
    result = run_surface_renewal(df, cfg=config)

    assert len(result) >= 1
    assert np.isfinite(result["H_uncal"]).all()
    assert (result["tau_star"] > 0).all()
    assert np.isfinite(result["S3_tau"]).all()

    # rotation="none" should also run without raising.
    config_none = PipelineConfig(fs=10, method="chen97", rotation="none", block="30min")
    run_surface_renewal(df, cfg=config_none)


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
