# src/surface_renewal/compute.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import pandas as pd

from .pipeline import PipelineConfig, run_surface_renewal


MethodName = Literal["snyder", "chen97"]
RotationMode = Literal["planar_fit", "double", "none"]
DespikeMethod = Literal["hampel", "gaussian"]


@dataclass
class ComputeConfig:
    """Back-compat wrapper around :class:`PipelineConfig`.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz (e.g., 10 or 20).
    z : float
        Measurement height (m), i.e., the V/A volume-to-area ratio. Required by the
        Snyder ideal-SR flux so that H has units of W m⁻².
    alpha : float, default 1.0
        Dimensionless SR weighting factor (~0.5–1.0); 1.0 is "ideal SR".
    block : str, default "30min"
        Time-averaging period for SR fluxes.
    method : {"snyder","chen97"}, default "snyder"
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
    d : float, default 0.0
        Zero-plane displacement height (m). Chen97 geometric scaling.
    h : float or None, default None
        Canopy height (m). Chen97 roughness-sublayer scaling.
    z_star : float or None, default None
        Roughness-sublayer top (m). Chen97; defaults to ``h`` (or 0.0).
    a_comb : float, default 0.4
        Chen97 combined coefficient ``alpha*beta**(2/3)*gamma`` (~0.4).
    """
    fs: float
    z: float
    alpha: float = 1.0
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

    # Chen97 geometric-scaling parameters
    d: float = 0.0
    h: Optional[float] = None
    z_star: Optional[float] = None
    a_comb: float = 0.4

    def to_pipeline_config(self) -> PipelineConfig:
        """Translate to the canonical :class:`PipelineConfig`."""
        return PipelineConfig(
            fs=self.fs,
            z=self.z,
            alpha=self.alpha,
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
            d=self.d,
            h=self.h,
            z_star=self.z_star,
            a_comb=self.a_comb,
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
    p.add_argument("--z", type=float, required=True, help="Measurement height (m), the V/A volume-to-area ratio.")
    p.add_argument("--alpha", type=float, default=1.0, help="SR weighting factor (default 1.0 = ideal SR).")
    p.add_argument("--block", default="30min", help="Block period, e.g. 30min.")
    p.add_argument("--method", choices=["snyder", "chen97"], default="snyder")
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

    p.add_argument("--d", type=float, default=0.0, help="Zero-plane displacement height (m); Chen97.")
    p.add_argument("--h", type=float, default=None, help="Canopy height (m); Chen97 roughness sublayer.")
    p.add_argument("--z-star", type=float, default=None, help="Roughness-sublayer top (m); Chen97.")
    p.add_argument("--a-comb", type=float, default=0.4, help="Chen97 combined coefficient (~0.4).")

    p.add_argument("--time-col", default=None, help="Name of timestamp column if needed.")
    p.add_argument("--out", default=None, help="Optional output Parquet/CSV path.")
    return p


def _to_cfg(ns) -> ComputeConfig:
    return ComputeConfig(
        fs=ns.fs,
        z=ns.z,
        alpha=ns.alpha,
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
        d=ns.d,
        h=ns.h,
        z_star=ns.z_star,
        a_comb=ns.a_comb,
    )


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point: `python -m surface_renewal.compute ...`"""
    import sys
    from .io import write_flux_timeseries, read_highfreq

    parser = _build_argparser()
    ns = parser.parse_args(argv)

    cfg = _to_cfg(ns)
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
