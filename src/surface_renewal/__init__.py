"""Surface renewal analysis package."""

from . import most
from .pipeline import run_surface_renewal, PipelineConfig
from .compute import compute, ComputeConfig
from .preprocess.calibration import Calibration
from .methods.snyder import estimate_H_snyder, SnyderResult
from .methods.chen97 import estimate_H_chen, ChenResult
from .methods.fvs import estimate_H_fvs, FVSResult
from .methods.castellvi import estimate_H_castellvi, CastellviResult
from .methods.wavelet import detect_ramps_wavelet, WaveletRampResult
from .methods.analysis import compare_methods, method_agreement
from .structure import structure_functions, pick_optimal_lag, estimate_CT2

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
    "estimate_H_fvs",
    "FVSResult",
    "estimate_H_castellvi",
    "CastellviResult",
    "detect_ramps_wavelet",
    "WaveletRampResult",
    "compare_methods",
    "method_agreement",
    "structure_functions",
    "pick_optimal_lag",
    "estimate_CT2",
    "most",
    "__version__",
]
