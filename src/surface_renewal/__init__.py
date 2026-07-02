"""Surface renewal analysis package."""

from .pipeline import run_surface_renewal, PipelineConfig
from .compute import compute, ComputeConfig
from .preprocess.calibration import Calibration
from .methods.snyder import estimate_H_snyder, SnyderResult
from .methods.chen97 import estimate_H_chen, ChenResult
from .structure import structure_functions, pick_optimal_lag

__version__ = "0.1.0"

__all__ = [
    "run_surface_renewal",
    "PipelineConfig",
    "compute",
    "ComputeConfig",
    "Calibration",
    "estimate_H_snyder",
    "SnyderResult",
    "estimate_H_chen",
    "ChenResult",
    "structure_functions",
    "pick_optimal_lag",
    "__version__",
]
