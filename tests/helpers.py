import numpy as np
import pandas as pd

def generate_test_data(fs: int, n_rows: int, realistic: bool = False) -> pd.DataFrame:
    """Generate a sample DataFrame for testing."""
    # Start at a round number for easy grouping
    start_time = pd.to_datetime("2023-01-01 00:00:00")
    # Use a high-frequency DatetimeIndex
    time = np.arange(n_rows) / fs
    time_index = pd.to_datetime(
        time,
        unit="s",
        origin=start_time,
    )

    if realistic:
        # More realistic data to avoid SVD errors
        u_mean = 2.5
        v_mean = 0.1
        w_mean = 0.0
        u_std = 0.5
        v_std = 0.5
        w_std = 0.1

        # Introduce some correlation and sinusoidal variation
        u = u_mean + np.random.randn(n_rows) * u_std + 0.5 * np.sin(2 * np.pi * time / 60)
        v = v_mean + 0.5 * (u - u_mean) + np.random.randn(n_rows) * v_std + 0.2 * np.cos(2 * np.pi * time / 60)
        w = w_mean + 0.2 * (u - u_mean) + np.random.randn(n_rows) * w_std
    else:
        u = 2.5 + np.random.randn(n_rows) * 0.5
        v = 0.1 + np.random.randn(n_rows) * 0.5
        w = 0.0 + np.random.randn(n_rows) * 0.1

    data = {
        "T": 298.15 + np.random.randn(n_rows) * 0.5,  # Temperature in Kelvin
        "u": u,
        "v": v,
        "w": w,
        "Rn": 500 + np.random.randn(n_rows) * 50,     # Net radiation in W/m^2
        "G": 50 + np.random.randn(n_rows) * 10,       # Ground heat flux in W/m^2
    }
    return pd.DataFrame(data, index=time_index)
