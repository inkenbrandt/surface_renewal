# src/surface_renewal/compute.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import pandas as pd

from .pipeline import PipelineConfig, run_surface_renewal


MethodName = Literal["snyder", "chen97", "fvs", "castellvi", "wavelet"]
RotationMode = Literal["planar_fit", "double", "none"]
DespikeMethod = Literal["hampel", "gaussian"]


@dataclass
class ComputeConfig:
    """Back-compat wrapper around :class:`PipelineConfig`.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz (e.g., 10 or 20).
    block : str, default "30min"
        Time-averaging period for SR fluxes.
    method : {"snyder","chen97","fvs","castellvi","wavelet"}, default "snyder"
        SR method used to compute uncalibrated H.
    rotation : {"planar_fit","double","none"}, default "planar_fit"
        Wind rotation scheme.
    despike_method : {"hampel","gaussian"}, default "hampel"
        Spike detector for T/u/v/w.
    hampel_window : int, default 11
        Hampel window length (samples).
    hampel_sigmas : float, default 3.0
        MAD threshold multiplier for Hampel.
    gaussian_nw : int, default 201
        Gaussian despiker window (samples).
    gaussian_sig : float, default 4.0
        Sigma threshold for gaussian despiker.
    gaussian_buffer : int, default 3
        Symmetric buffer (samples) around spikes → NaN.
    interp_kind : {"linear","nearest","cubic",None}, default None
        Optional interpolation for short gaps after despiking.
    interp_max_gap : int or None, default None
        Max consecutive NaNs (samples) to interpolate.
    stability_ustar : float, default 0.05
    stability_relS3 : float, default 1e-3
    stability_stdT : float, default 0.02
    daytime_only : bool, default False
        If True and Rn is available, require Rn>0 to accept block.
    z_m : float, optional
        Measurement height above the zero-plane displacement (``z_sensor - d``,
        with ``d ≈ 0.66 * canopy_height``), in metres. Required by
        height-dependent methods (``fvs``, ``castellvi``).
    free_convection_fallback : bool, default False
        Enable the free-convection fallback for strongly unstable, low-wind
        blocks (records the choice in the ``flux_method_used`` column). Requires
        ``z_m``. See :class:`~surface_renewal.pipeline.PipelineConfig`.
    fc_ustar_max : float, default 0.1
        Upper u* bound (m s⁻¹) for the fallback.
    fc_zeta_max : float, default -0.5
        Upper zeta bound; fallback applies when zeta < this.
    """
    fs: float
    block: str = "30min"
    method: MethodName = "snyder"
    rotation: RotationMode = "planar_fit"

    despike_method: DespikeMethod = "hampel"
    hampel_window: int = 11
    hampel_sigmas: float = 3.0
    gaussian_nw: int = 201
    gaussian_sig: float = 4.0
    gaussian_buffer: int = 3
    interp_kind: Optional[Literal["linear", "nearest", "cubic"]] = None
    interp_max_gap: Optional[int] = None

    stability_ustar: float = 0.05
    stability_relS3: float = 1e-3
    stability_stdT: float = 0.02
    daytime_only: bool = False
    z_m: Optional[float] = None

    free_convection_fallback: bool = False
    fc_ustar_max: float = 0.1
    fc_zeta_max: float = -0.5

    def to_pipeline_config(self) -> PipelineConfig:
        """Translate to the canonical :class:`PipelineConfig`."""
        return PipelineConfig(
            fs=self.fs,
            block=self.block,
            method=self.method,
            rotation=self.rotation,
            despike_method=self.despike_method,
            hampel_window=self.hampel_window,
            hampel_sigmas=self.hampel_sigmas,
            gaussian_nw=self.gaussian_nw,
            gaussian_sig=self.gaussian_sig,
            gaussian_buffer=self.gaussian_buffer,
            interp_kind=self.interp_kind,
            interp_max_gap=self.interp_max_gap,
            stability_ustar=self.stability_ustar,
            stability_relS3=self.stability_relS3,
            stability_stdT=self.stability_stdT,
            daytime_only=self.daytime_only,
            z_m=self.z_m,
            free_convection_fallback=self.free_convection_fallback,
            fc_ustar_max=self.fc_ustar_max,
            fc_zeta_max=self.fc_zeta_max,
        )


def compute(
    data: pd.DataFrame | str,
    *,
    cfg: ComputeConfig,
    time_col: Optional[str] = None,
) -> pd.DataFrame:
    """Back-compat entry that just calls :func:`pipeline.run_surface_renewal`.

    Parameters
    ----------
    data : pd.DataFrame or str
        High-frequency table (or path to CSV/Parquet) with at least
        ``["T","u","v","w"]`` and optionally ``["Rn","G"]``.
    cfg : ComputeConfig
        Wrapper configuration (converted to :class:`PipelineConfig`).
    time_col : str, optional
        Name of timestamp column if `data` lacks a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Block-level outputs from the SR pipeline, including uncalibrated H.
    """
    return run_surface_renewal(data, cfg=cfg.to_pipeline_config(), time_col=time_col)


# ------------------------------ CLI support -------------------------------- #

def _build_argparser():
    import argparse
    p = argparse.ArgumentParser(
        prog="surface_renewal.compute",
        description="Run Surface Renewal pipeline (thin wrapper around pipeline.run_surface_renewal).",
    )
    p.add_argument("input", help="Path to CSV/Parquet with high-frequency data.")
    p.add_argument("--fs", type=float, required=True, help="Sampling frequency (Hz).")
    p.add_argument("--block", default="30min", help="Block period, e.g. 30min.")
    p.add_argument("--method", choices=["snyder", "chen97", "fvs", "castellvi", "wavelet"], default="snyder")
    p.add_argument("--rotation", choices=["planar_fit", "double", "none"], default="planar_fit")

    p.add_argument("--despike", choices=["hampel", "gaussian"], default="hampel")
    p.add_argument("--hampel-window", type=int, default=11)
    p.add_argument("--hampel-sigmas", type=float, default=3.0)
    p.add_argument("--gauss-nw", type=int, default=201)
    p.add_argument("--gauss-sig", type=float, default=4.0)
    p.add_argument("--gauss-buffer", type=int, default=3)
    p.add_argument("--interp", choices=["linear", "nearest", "cubic"], default=None)
    p.add_argument("--interp-max-gap", type=int, default=None)

    p.add_argument("--min-ustar", type=float, default=0.05)
    p.add_argument("--min-relS3", type=float, default=1e-3)
    p.add_argument("--min-stdT", type=float, default=0.02)
    p.add_argument("--daytime-only", action="store_true")

    p.add_argument("--z-m", type=float, default=None,
                   help="Measurement height above zero-plane displacement (m); z_sensor - d, d≈0.66*canopy height.")

    p.add_argument("--free-convection-fallback", action="store_true",
                   help="Fall back to the free-convection H estimate on strongly "
                        "unstable, low-wind blocks (requires --z-m).")
    p.add_argument("--fc-ustar-max", type=float, default=0.1,
                   help="Upper u* bound (m/s) for the free-convection fallback.")
    p.add_argument("--fc-zeta-max", type=float, default=-0.5,
                   help="Fallback applies when block zeta < this value.")

    p.add_argument("--time-col", default=None, help="Name of timestamp column if needed.")
    p.add_argument("--out", default=None, help="Optional output Parquet/CSV path.")

    p.add_argument("--compare", action="store_true",
                   help="Run all SR methods on the same data and print the "
                        "pairwise method-agreement table (ignores --method). "
                        "Height-dependent methods need --z-m.")
    return p


def _to_cfg(ns) -> ComputeConfig:
    return ComputeConfig(
        fs=ns.fs,
        block=ns.block,
        method=ns.method,
        rotation=ns.rotation,
        despike_method=ns.despike,
        hampel_window=ns.hampel_window,
        hampel_sigmas=ns.hampel_sigmas,
        gaussian_nw=ns.gauss_nw,
        gaussian_sig=ns.gauss_sig,
        gaussian_buffer=ns.gauss_buffer,
        interp_kind=ns.interp,
        interp_max_gap=ns.interp_max_gap,
        stability_ustar=ns.min_ustar,
        stability_relS3=ns.min_relS3,
        stability_stdT=ns.min_stdT,
        daytime_only=ns.daytime_only,
        z_m=ns.z_m,
        free_convection_fallback=ns.free_convection_fallback,
        fc_ustar_max=ns.fc_ustar_max,
        fc_zeta_max=ns.fc_zeta_max,
    )


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point: `python -m surface_renewal.compute ...`"""
    import sys
    from .io import write_flux_timeseries, read_highfreq

    parser = _build_argparser()
    ns = parser.parse_args(argv)

    cfg = _to_cfg(ns)

    if ns.compare:
        # Run every SR method on the same data and report where they agree.
        from .methods.analysis import compare_methods, method_agreement

        wide = compare_methods(
            ns.input, cfg=cfg.to_pipeline_config(), time_col=ns.time_col,
        )
        agree = method_agreement(wide)
        with pd.option_context("display.max_columns", None, "display.width", 160):
            print("Per-block method comparison (tail):")
            print(wide.tail(10))
            print("\nPairwise method agreement:")
            print(agree)
        if ns.out:
            write_flux_timeseries(wide, ns.out, metadata={"mode": "compare", "block": cfg.block})
        return 0

    # Let pipeline handle reading; we could also pre-read to validate here.
    out = compute(ns.input, cfg=cfg, time_col=ns.time_col)

    if ns.out:
        write_flux_timeseries(out, ns.out, metadata={"method": cfg.method, "block": cfg.block})
    else:
        # Print a small preview to stdout
        with pd.option_context("display.max_columns", None, "display.width", 160):
            print(out.tail(10))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
