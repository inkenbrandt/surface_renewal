"""General SR analysis utilities (merges analysis_strfnc.py)."""
from __future__ import annotations
import numpy as np
import pandas as pd

def detect_ramps(T: np.ndarray, fs: int) -> dict:
    """Detect ramp events and return stats (amplitude, duration, counts)."""
    # TODO
    return {{"amp": [], "tau": []}}