# src/surface_renewal/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any, Iterable

import numpy as np
import pandas as pd

# Local imports
from .io import read_highfreq
from .preprocess.despike import despike_dataframe, velocity_temperature_consistency
from .preprocess.rotation import planar_fit, double_rotation, friction_velocity, RotationResult
from .preprocess.stability import compute_block_diagnostics, stability_ok, BlockDiagnostics
from .preprocess.calibration import rho_air_ideal, cp_air_const, Calibration  # optional helpers
from .structure import estimate_CT2
from .most import obukhov_length

# Methods
from .methods.snyder import estimate_H_snyder, SnyderResult  # Snyder96 cubic-ramp  :contentReference[oaicite:3]{index=3}
from .methods.chen97 import estimate_H_chen, ChenResult       # Chen97 variant       :contentReference[oaicite:4]{index=4}
from .methods.fvs import estimate_H_fvs, FVSResult, estimate_H_free_convection  # flux–variance similarity
from .methods.castellvi import estimate_H_castellvi, CastellviResult  # Castellví04 calibration-free
from .methods.wavelet import detect_ramps_wavelet, WaveletRampResult  # Collineau & Brunet 1993


MethodName = Literal["snyder", "chen97", "fvs", "castellvi", "wavelet"]


@dataclass
class PipelineConfig:
    """Configuration for the SR pipeline.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz (e.g., 10 or 20).
    block : str, default "30min"
        Time-averaging period for SR fluxes (pandas offset alias).
    method : {"snyder", "chen97", "fvs", "castellvi", "wavelet"}, default "snyder"
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
    z_m : float, optional
        Measurement height of the sonic/thermocouple above the zero-plane
        displacement, in metres. This should be ``z_sensor - d``, where the
        zero-plane displacement ``d ≈ 0.66 * canopy_height``. Methods that
        require it (``fvs``, ``castellvi``) will raise a ``ValueError`` at
        config time if it is left as ``None``.
    free_convection_fallback : bool, default False
        If True, strongly unstable, low-wind blocks fall back to the
        free-convection estimate :func:`~surface_renewal.methods.fvs.\
estimate_H_free_convection` when the primary :math:`u_*`-based method
        (``chen97``, ``fvs``, ``castellvi``) is expected to degrade. The
        substitution happens per block in :func:`_compute_block_flux`; the
        chosen estimate is recorded in the ``flux_method_used`` output column.
        Requires ``z_m`` to be set.
    fc_ustar_max : float, default 0.1
        Upper :math:`u_*` bound (m s⁻¹) for the fallback: it is only considered
        when the block ``ustar`` is below this value.
    fc_zeta_max : float, default -0.5
        Upper stability bound: the fallback is only applied when the block's
        :math:`\\zeta = z_m/L` (from ``obukhov_length`` of the primary
        :math:`H`) is *below* this (i.e. sufficiently unstable).
    """
    fs: float
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
    z_m: Optional[float] = None   # sonic/thermocouple height above
                                  # zero-plane displacement (m)

    free_convection_fallback: bool = False
    fc_ustar_max: float = 0.1     # m/s
    fc_zeta_max: float = -0.5     # apply when zeta < this


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
    # Quick sanity flagging (non-destructive); True marks suspicious records.
    qc_range_flag = velocity_temperature_consistency(df)

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

    # Rotate winds according to the configured scheme; all branches produce
    # u_r, v_r, w_r columns downstream code relies on.
    if cfg.rotation == "planar_fit":
        # Planar-fit (stable across terrain); stores matrix in attrs
        rot: RotationResult = planar_fit(df_d)
        out = rot.df  # contains u_r, v_r, w_r
    elif cfg.rotation == "double":
        # Double rotation over the full dataset; adds u_r, v_r, w_r
        out = double_rotation(df_d, by="none")
    elif cfg.rotation == "none":
        # No rotation: rotated columns are just the raw components (identity R)
        out = df_d.copy()
        out["u_r"] = out["u"]
        out["v_r"] = out["v"]
        out["w_r"] = out["w"]
        rot = RotationResult(df=out, R=np.eye(3), meta={"rotation": "none"})
        out = rot.df
    else:
        raise ValueError(f"Unknown rotation scheme: {cfg.rotation!r}")

    # Carry the QC screen through as a boolean column aligned to the output.
    out["qc_range_flag"] = qc_range_flag.reindex(out.index).fillna(False).astype(bool)
    return out


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
    diag.passed = passed

    # Compute uncalibrated H via selected SR method
    zeta = np.nan       # only height-dependent methods (fvs, castellvi) fill this in
    alpha_sr = np.nan   # Castellví analytic weighting factor; NaN for other methods
    n_ramps = 0         # only the wavelet method reports a ramp count
    if cfg.method == "snyder":
        sres: SnyderResult = estimate_H_snyder(
            grp["T"].to_numpy(float), hz=hz, rho=rho, cp=cp
        )  # Snyder uses S2/S3/S5 + Cardano; no u* term  :contentReference[oaicite:5]{index=5}
        H_uncal = sres.H
        tau_star = sres.tau
        dt_opt = sres.dt_opt
        S3_tau = np.nan  # different construction; supplied by diag below if needed
    elif cfg.method == "fvs":
        if cfg.z_m is None:
            raise ValueError("method='fvs' requires cfg.z_m")
        # Block-mean temperature in Kelvin (accept Celsius or Kelvin inputs).
        T_mean = float(np.nanmean(grp["T"].to_numpy(float)))
        T_K_mean = T_mean + 273.15 if T_mean < 150.0 else T_mean
        fres: FVSResult = estimate_H_fvs(
            sigma_T=diag.stdT,
            ustar=diag.u_star,
            T_K=T_K_mean,
            z_m=cfg.z_m,
            rho=rho, cp=cp,
            sign_hint=diag.S3_tau,  # sign(S3(τ*)) supplies the flux direction
        )
        H_uncal = fres.H
        tau_star = diag.tau_opt
        dt_opt = diag.tau_opt
        S3_tau = diag.S3_tau
        zeta = fres.zeta
    elif cfg.method == "castellvi":
        if cfg.z_m is None:
            raise ValueError("method='castellvi' requires cfg.z_m")
        # Block-mean temperature in Kelvin (accept Celsius or Kelvin inputs).
        T_mean = float(np.nanmean(grp["T"].to_numpy(float)))
        T_K_mean = T_mean + 273.15 if T_mean < 150.0 else T_mean
        cvres: CastellviResult = estimate_H_castellvi(
            grp["T"].to_numpy(float),
            hz=hz,
            ustar=diag.u_star,
            T_K=T_K_mean,
            z_m=cfg.z_m,
            rho=rho, cp=cp,
        )  # calibration-free: analytic alpha from SR + MOST (ramp sign gives H sign)
        H_uncal = cvres.H
        tau_star = cvres.tau
        dt_opt = cvres.tau
        S3_tau = diag.S3_tau
        zeta = cvres.zeta
        alpha_sr = cvres.alpha
    elif cfg.method == "wavelet":
        wres: WaveletRampResult = detect_ramps_wavelet(
            grp["T"].to_numpy(float), hz=hz, rho=rho, cp=cp,
        )  # wavelet ramp detection: H = rho*cp*A/tau (uncalibrated)
        H_uncal = wres.H
        tau_star = wres.tau
        dt_opt = wres.tau
        S3_tau = np.nan
        n_ramps = wres.n_ramps
    else:
        cres: ChenResult = estimate_H_chen(
            T=grp["T"].to_numpy(float),
            u=grp["u"].to_numpy(float),
            v=grp["v"].to_numpy(float),
            w=grp["w"].to_numpy(float),
            hz=hz, rho=rho, cp=cp,
            rotation=("planar_fit" if cfg.rotation == "planar_fit" else
                      "double" if cfg.rotation == "double" else "none"),
        )  # Chen uses S3(τ*), u*, τ*  :contentReference[oaicite:6]{index=6}
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

    # Friction velocity (from rotated covariances added by _preprocess_df).
    # `friction_velocity(..., method="global")` returns a length-1 Series; take
    # the scalar via .iloc[0] (pandas 3.x disallows float() on a Series).
    ustar_val = float(
        friction_velocity(grp, u_col="u_r", v_col="v_r", w_col="w_r", method="global").iloc[0]
    )

    # Free-convection fallback for strongly unstable, low-wind blocks, where the
    # u*-based methods (chen97, fvs, castellvi) degrade as u* -> 0. When the flag
    # is set and the primary method is u*-based, check whether this block is both
    # low-wind (ustar < fc_ustar_max) and sufficiently unstable: zeta is computed
    # from obukhov_length of the primary H, so the test is consistent across
    # methods (including chen97, which does not otherwise fill zeta).
    flux_method_used = "primary"
    if (
        cfg.free_convection_fallback
        and cfg.method in ("chen97", "fvs", "castellvi")
        and np.isfinite(H_uncal)
        and ustar_val < cfg.fc_ustar_max
    ):
        if cfg.z_m is None:
            raise ValueError("free_convection_fallback requires cfg.z_m")
        T_mean = float(np.nanmean(grp["T"].to_numpy(float)))
        T_K_mean = T_mean + 273.15 if T_mean < 150.0 else T_mean
        L_sw = obukhov_length(ustar_val, T_K_mean, H_uncal, rho=rho, cp=cp)
        zeta_sw = cfg.z_m / L_sw if np.isfinite(L_sw) else np.nan
        # Free convection implies H > 0. Only substitute when the primary
        # estimate is itself positive; if the primary H was negative (stable /
        # downward flux), keep it rather than force a spurious positive flux.
        if np.isfinite(zeta_sw) and zeta_sw < cfg.fc_zeta_max and H_uncal > 0.0:
            H_uncal = estimate_H_free_convection(
                sigma_T=diag.stdT, T_K=T_K_mean, z_m=cfg.z_m, rho=rho, cp=cp,
            )
            flux_method_used = "free_convection"
        # Report the switch-diagnostic zeta when the primary method left it NaN.
        if not np.isfinite(zeta):
            zeta = zeta_sw

    # Fraction of this block's records flagged by the QC range screen.
    frac_flagged = grp["qc_range_flag"].mean()

    # Block-mean horizontal wind speed (needed by height-dependent methods).
    U = float(np.nanmean(np.hypot(grp["u"], grp["v"])))

    # Temperature structure parameter C_T^2 (Wyngaard et al. 1971) via the
    # inertial-subrange second-order structure function. Near-calm blocks
    # (U <= 0.1 m/s) yield NaN rather than being skipped.
    CT2, CT2_r2 = estimate_CT2(grp["T"].to_numpy(float), hz=hz, U=U)

    return {
        "passed": bool(passed),
        "H_uncal": float(H_uncal),
        "LE_resid": float(LE),
        "ustar": ustar_val,
        "U_mean": U,
        "tau_star": float(tau_star),
        "dt_opt": float(dt_opt),
        "n_ramps": int(n_ramps),
        "zeta": float(zeta) if np.isfinite(zeta) else np.nan,
        "alpha_sr": float(alpha_sr) if np.isfinite(alpha_sr) else np.nan,
        "S3_tau": float(S3_tau) if np.isfinite(S3_tau) else np.nan,
        "stdT": float(np.nanstd(grp["T"].to_numpy(float))),
        "rho": float(rho),
        "cp": float(cp),
        "frac_qc_flagged": float(frac_flagged),
        "CT2": float(CT2),
        "CT2_r2": float(CT2_r2),
        "flux_method_used": flux_method_used,
    }


def run_surface_renewal(
    data: pd.DataFrame | str | None = None,
    *,
    cfg: PipelineConfig,
    time_col: Optional[str] = None,
    alpha: Optional[float | Calibration] = None,
    preprocessed: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """End-to-end surface-renewal pipeline → block-level fluxes & diagnostics.

    Parameters
    ----------
    data : pd.DataFrame or str, optional
        Either a preloaded high-frequency DataFrame with columns at least
        ``["T","u","v","w"]`` (optionally ``["Rn","G"]``), or a path to a CSV/Parquet file.
        If a path is provided, `read_highfreq` will be used. May be ``None`` only
        when ``preprocessed`` is supplied.
    cfg : PipelineConfig
        Configuration for sampling frequency, despiking, rotation, method, screening, etc.
    time_col : str, optional
        If `data` is a table without a DatetimeIndex, name of the timestamp column.
    preprocessed : pd.DataFrame, optional
        A DataFrame already run through :func:`_ensure_df` + :func:`_preprocess_df`
        (i.e. despiked, rotated, carrying ``u_r``/``v_r``/``w_r`` and
        ``qc_range_flag``). When supplied, the read + preprocess stages are
        skipped and this frame is used directly. This lets callers such as
        :func:`surface_renewal.methods.analysis.compare_methods` preprocess once
        and evaluate several methods on the identical cleaned series without
        repeating the expensive despike/rotation work. When given, ``data`` and
        ``time_col`` are ignored.
    alpha : float or Calibration, optional
        Optional block-scale calibration applied to ``H_uncal`` to produce a
        calibrated sensible heat flux ``H_cal``:

        - If a ``float``, ``H_cal = alpha * H_uncal``.
        - If a :class:`~surface_renewal.preprocess.calibration.Calibration`,
          ``H_cal = alpha.apply(H_uncal)`` (i.e. ``alpha * H_uncal + beta``).
        - If ``None`` (default), no ``H_cal`` column is added (backwards compatible).

        When ``H_cal`` is present and both ``Rn`` and ``G`` are available, a
        calibrated residual latent heat flux ``LE_cal = Rn - G - H_cal`` is also added.

    Returns
    -------
    pd.DataFrame
        Block-level results indexed by block end time with columns:
        ``["H_uncal","LE_resid","passed","ustar","tau_star","dt_opt","S3_tau","stdT","rho","cp"]``.
        If ``alpha`` is provided, an ``H_cal`` column is added (and ``LE_cal`` when
        ``Rn`` and ``G`` are available).

    Notes
    -----
    - **Uncalibrated H** is returned by design; pass ``alpha`` (or fit and apply a
      block-scale **alpha** using your reference EC H via
      `Calibration.from_reference(...).apply(...)`) to obtain calibrated H. :contentReference[oaicite:7]{index=7}
    - Snyder method uses S2/S3/S5 + Cardano cubic recovery; Chen97 uses S3(τ*), u*,
      and τ* scaling; both follow the implementations in `methods/`. :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}
    """
    # 0) + 1) Obtain the despiked/rotated frame, reusing a caller-supplied one
    # when available so preprocessing is not repeated across methods.
    if preprocessed is not None:
        df_prep = preprocessed
    else:
        if data is None:
            raise ValueError("run_surface_renewal requires `data` when `preprocessed` is None")
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
        cols = ["H_uncal", "LE_resid", "passed", "ustar", "U_mean", "tau_star", "dt_opt", "n_ramps", "zeta", "alpha_sr", "S3_tau", "stdT", "rho", "cp", "frac_qc_flagged", "CT2", "CT2_r2", "flux_method_used"]
        if alpha is not None:
            cols.append("H_cal")
            cols.append("LE_cal")
        return pd.DataFrame(columns=cols)

    # 3) Assemble tidy frame
    idx = pd.to_datetime([ix for ix, _ in rows])
    out = pd.DataFrame([r for _, r in rows], index=idx).sort_index()

    # 4) Optional calibration → H_cal (and residual LE_cal when Rn, G available)
    if alpha is not None:
        if isinstance(alpha, Calibration):
            out["H_cal"] = alpha.apply(out["H_uncal"]).to_numpy()
        else:
            out["H_cal"] = out["H_uncal"] * float(alpha)

        # Recompute the residual LE using calibrated H when radiation terms exist.
        # LE_resid was built from Rn - G - H_uncal, so Rn - G = LE_resid + H_uncal.
        if "Rn" in df_prep.columns and "G" in df_prep.columns:
            out["LE_cal"] = out["LE_resid"] + out["H_uncal"] - out["H_cal"]

    return out
