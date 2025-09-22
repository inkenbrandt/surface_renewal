from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class Calibration:
    alpha: float = 1.0

    @classmethod
    def from_reference(cls, H_model: pd.Series, H_ref: pd.Series) -> "Calibration":
        """Calibrate alpha by robust slope fitting (no intercept)."""
        df = pd.concat({"x": H_model, "y": H_ref}, axis=1).dropna()
        if len(df) < 3:
            return cls(alpha=1.0)
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()
        # Constrained through origin: alpha = sum(x*y) / sum(x^2)
        denom = float(np.dot(x, x))
        if denom == 0:
            return cls(alpha=1.0)
        alpha = float(np.dot(x, y) / denom)
        return cls(alpha=alpha)

    def apply(self, H_series: pd.Series) -> pd.Series:
        return self.alpha * H_series
