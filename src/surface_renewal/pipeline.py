# src/surface_renewal/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any, Iterable

import numpy as np
import pandas as pd

# Local imports
from .io import read_highfreq
from .preprocess.despike import despike_dataframe, velocity_temperature_consistency
from .preprocess.rotation import planar_fit, friction_velocity, RotationResult
from .preprocess.stability import compute_block_diagnostics, stability_ok, BlockDiagnostics
from .preprocess.calibration import rho_air_ideal, cp_air_const  # optional helpers

# Methods
from .methods.snyder import estimate_H_snyder, SnyderResult  # Snyder96 cubic-ramp  :contentReference[oaicite:3]{index=3}
from .methods.chen97 import estimate_H_chen, ChenResult       # Chen97 variant       :contentReference[oaicite:4]{index=4}


MethodName = Literal["snyder", "chen97"]


@dataclass
class PipelineConfig:
    """Configuration for the SR pipeline.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz (e.g., 10 or 20).
    z : float
        Measurement height (m), i.e., the V/A volume-to-area ratio. Required by the
        Snyder ideal-SR flux so that H has units of W m⁻² rather than W m⁻³.
    alpha : float, default 1.0
        Dimensionless SR weighting factor (~0.5–1.0); 1.0 is "ideal SR".
    block : str, default "30min"
        Time-averaging period for SR fluxes (pandas offset alias).
    method : {"snyder", "chen97"}, default "snyder"
        Surface-renewal method to compute uncalibrated H.
    rotation : {"planar_fit", "double", "none"}, default "planar_fit"
        Wind rotation scheme used for diagnostics/u* (Chen97) and consistency.
    despike_method : {"hampel", "gaussian"}, default "hampel"
        Detector used in `despike_dataframe`.
    hampel_window : int, default 11
        Window length (samples) for Hampel.
    hampel_sigmas : float, default 3.0
        MAD threshold multiplier for Hampel.
    gaussian_nw : int, default 201
        Window length (samples) for gaussian despiker.
    gaussian_sig : float, default 4.0
        Sigma threshold for gaussian despiker.
    gaussian_buffer : int, default 3
        Symmetric buffer (samples) around flagged spikes → NaN.
    interp_kind : {"linear","nearest","cubic",None}, default None
        If set, interpolate short spike gaps after detection.
    interp_max_gap : int or None, default None
        Max consecutive NaNs (samples) to interpolate (guard against long gaps).
    stability_ustar : float, default 0.05
        Minimum u* (m s⁻¹) to accept a block.
    stability_relS3 : float, default 1e-3
        Minimum |S3(τ*)| / std(T)^3 (dimensionless) to accept a block.
    stability_stdT : float, default 0.02
        Minimum std(T) (K) to accept a block.
    daytime_only : bool, default False
        If True, require Rn>0 (only matters if Rn provided).
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
    rotation: Literal["planar_fit", "double", "none"] = "planar_fit"

    despike_method: Literal["hampel", "gaussian"] = "hampel"
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


def _ensure_df(
    data: pd.DataFrame | str,
    *,
    fs: float,
    time_col: Optional[str] = None,
) -> pd.DataFrame:
    """Return a datetime-indexed DataFrame with T, u, v, w columns."""
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = read_highfreq(data, freq_hz=int(fs))

    # Try to ensure required columns exist
    required = ["T", "u", "v", "w"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Make datetime index if not already present
    if not isinstance(df.index, pd.DatetimeIndex):
        if time_col and (time_col in df.columns):
            df = df.set_index(pd.to_datetime(df[time_col], errors="coerce"))
        else:
            # As a fallback, build a synthetic index using fs
            start = pd.Timestamp("1970-01-01")
            dt = pd.to_timedelta(np.arange(len(df)) / fs, unit="s")
            df.index = start + dt
    return df.sort_index()


def _preprocess_df(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Despike & rotate winds; return DataFrame with u_r, v_r, w_r and cleaned T."""
    # Quick sanity flagging (non-destructive)
    _ = velocity_temperature_consistency(df)

    # Despike
    df_d, _masks = despike_dataframe(
        df, columns=["T", "u", "v", "w"],
        method=cfg.despike_method,
        window=cfg.hampel_window,
        n_sigmas=cfg.hampel_sigmas,
        nw=cfg.gaussian_nw,
        sig=cfg.gaussian_sig,
        buffer=cfg.gaussian_buffer,
        interpolate=cfg.interp_kind,
        max_gap=cfg.interp_max_gap,
    )

    # Planar-fit by default (stable across terrain); store matrix in attrs
    rot: RotationResult = planar_fit(df_d)
    return rot.df  # contains u_r, v_r, w_r


def _block_iter(df: pd.DataFrame, block: str) -> Iterable[pd.DataFrame]:
    """Yield block-sized dataframes with at least a few seconds of data."""
    for _, grp in df.groupby(pd.Grouper(freq=block)):
        if len(grp) == 0:
            continue
        yield grp


def _compute_block_flux(
    grp: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    rho: Optional[float] = None,
    cp: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute diagnostics and uncalibrated fluxes for a single block."""
    hz = cfg.fs

    # Density & cp (optional; use helpers if not provided)
    T_K = grp["T"].to_numpy(float)
    if rho is None:
        rho = float(rho_air_ideal(T_K).median(skipna=True))
    if cp is None:
        cp = float(cp_air_const(T_K).median(skipna=True))

    # Stability diagnostics (uses u*, S3, τ*)
    diag: BlockDiagnostics = compute_block_diagnostics(
        T=grp["T"].to_numpy(float),
        u=grp["u"].to_numpy(float),
        v=grp["v"].to_numpy(float),
        w=grp["w"].to_numpy(float),
        hz=hz,
        rotation=("planar_fit" if cfg.rotation == "planar_fit" else
                  "double" if cfg.rotation == "double" else "none"),
    )

    # Optional daytime constraint if Rn is present
    Rn_blk = float(np.nanmean(grp["Rn"])) if "Rn" in grp.columns else None
    passed = stability_ok(
        diag,
        min_ustar=cfg.stability_ustar,
        min_rel_S3=cfg.stability_relS3,
        min_stdT=cfg.stability_stdT,
        daytime_only=cfg.daytime_only,
        Rn_block=Rn_blk,
    )

    # Compute uncalibrated H via selected SR method
    if cfg.method == "snyder":
        sres: SnyderResult = estimate_H_snyder(
            grp["T"].to_numpy(float), hz=hz, z=cfg.z, alpha=cfg.alpha, rho=rho, cp=cp
        )  # Snyder uses S2/S3/S5 + Cardano; no u* term  :contentReference[oaicite:5]{index=5}
        H_uncal = sres.H
        tau_star = sres.tau
        dt_opt = sres.dt_opt
        S3_tau = np.nan  # different construction; supplied by diag below if needed
    else:
        cres: ChenResult = estimate_H_chen(
            T=grp["T"].to_numpy(float),
            u=grp["u"].to_numpy(float),
            v=grp["v"].to_numpy(float),
            w=grp["w"].to_numpy(float),
            hz=hz, z=cfg.z, d=cfg.d, h=cfg.h, z_star=cfg.z_star,
            rho=rho, cp=cp, a_comb=cfg.a_comb,
            rotation=("planar_fit" if cfg.rotation == "planar_fit" else
                      "double" if cfg.rotation == "double" else "none"),
        )  # Chen (1997b) Eq.12: S3(r_m), u*^(2/3), geometric factor G
        H_uncal = cres.H
        tau_star = cres.tau_opt
        dt_opt = cres.tau_opt  # τ* is the selected lag here
        S3_tau = cres.S3_tau

    # Optionally compute LE by residual when radiation terms available
    LE = np.nan
    if ("Rn" in grp.columns) and ("G" in grp.columns):
        Rn_blk = float(np.nanmean(grp["Rn"]))
        G_blk = float(np.nanmean(grp["G"]))
        if np.isfinite(H_uncal):
            LE = Rn_blk - G_blk - H_uncal

    # Friction velocity from the rotated covariances produced in preprocessing.
    # The block already carries the rotated columns (u_r, v_r, w_r) from
    # planar_fit, so use them directly — renaming u/v/w to u_r/v_r/w_r here would
    # create duplicate column labels and break the lookup. `friction_velocity`
    # returns a length-1 Series for method="global"; take the scalar.
    ustar_val = float(friction_velocity(grp, method="global").iloc[0])

    return {
        "passed": bool(passed),
        "H_uncal": float(H_uncal),
        "LE_resid": float(LE),
        "ustar": ustar_val,
        "tau_star": float(tau_star),
        "dt_opt": float(dt_opt),
        "S3_tau": float(S3_tau) if np.isfinite(S3_tau) else np.nan,
        "stdT": float(np.nanstd(grp["T"].to_numpy(float))),
        "rho": float(rho),
        "cp": float(cp),
    }


def run_surface_renewal(
    data: pd.DataFrame | str,
    *,
    cfg: PipelineConfig,
    time_col: Optional[str] = None,
) -> pd.DataFrame:
    """End-to-end surface-renewal pipeline → block-level fluxes & diagnostics.

    Parameters
    ----------
    data : pd.DataFrame or str
        Either a preloaded high-frequency DataFrame with columns at least
        ``["T","u","v","w"]`` (optionally ``["Rn","G"]``), or a path to a CSV/Parquet file.
        If a path is provided, `read_highfreq` will be used.
    cfg : PipelineConfig
        Configuration for sampling frequency, despiking, rotation, method, screening, etc.
    time_col : str, optional
        If `data` is a table without a DatetimeIndex, name of the timestamp column.

    Returns
    -------
    pd.DataFrame
        Block-level results indexed by block end time with columns:
        ``["H_uncal","LE_resid","passed","ustar","tau_star","dt_opt","S3_tau","stdT","rho","cp"]``

    Notes
    -----
    - **Uncalibrated H** is returned by design; fit and apply a block-scale **alpha**
      using your reference EC H via `Calibration.from_reference(...).apply(...)` in a
      post-step. :contentReference[oaicite:7]{index=7}
    - Snyder method uses S2/S3/S5 + Cardano cubic recovery; Chen97 uses S3(τ*), u*,
      and τ* scaling; both follow the implementations in `methods/`. :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}
    """
    # 0) Ensure df and required columns
    df = _ensure_df(data, fs=cfg.fs, time_col=time_col)

    # 1) Preprocess (despike → rotation)
    df_prep = _preprocess_df(df, cfg)

    # 2) Compute per-block fluxes and diagnostics
    rows = []
    for grp in _block_iter(df_prep, cfg.block):
        if len(grp) < max(64, int(cfg.fs * 2)):
            continue
        res = _compute_block_flux(grp, cfg)
        rows.append((grp.index[-1], res))

    if not rows:
        return pd.DataFrame(
            columns=["H_uncal", "LE_resid", "passed", "ustar", "tau_star", "dt_opt", "S3_tau", "stdT", "rho", "cp"]
        )

    # 3) Assemble tidy frame
    idx = pd.to_datetime([ix for ix, _ in rows])
    out = pd.DataFrame([r for _, r in rows], index=idx).sort_index()
    return out
