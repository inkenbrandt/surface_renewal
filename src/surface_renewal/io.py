# src/surface_renewal/io.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable, Dict, Any
from pathlib import Path
import glob

import numpy as np
import pandas as pd

__all__ = [
    "SchemaError",
    "DEFAULT_ALIASES",
    "read_highfreq",
    "write_flux_timeseries",
    "infer_sampling_hz",
]


# --------------------------------------------------------------------------- #
# Errors & constants
# --------------------------------------------------------------------------- #

class SchemaError(ValueError):
    """Raised when required columns or index/time information are missing."""


# Common column aliases we’ll normalize to canonical names
DEFAULT_ALIASES: Dict[str, str] = {
    # temperature
    "temp": "T", "temperature": "T", "t": "T", "t_c": "T", "t_k": "T",
    # winds
    "u": "u", "v": "v", "w": "w",
    "ux": "u", "vy": "v", "wz": "w",
    "u_": "u", "v_": "v", "w_": "w",
    # radiation / soil
    "rn": "Rn", "rnet": "Rn", "netrad": "Rn",
    "g": "G", "soilheat": "G",
    # pressure
    "p": "P_Pa", "press": "P_Pa", "pressure": "P_Pa", "p_pa": "P_Pa",
    # time
    "time": "time", "timestamp": "time", "datetime": "time", "DateTime": "time",
}


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def _is_parquet(path: Path) -> bool:
    return path.suffix.lower() in (".parquet", ".pq", ".parq")

def _coerce_datetime_index(df: pd.DataFrame, time_col: Optional[str], tz: Optional[str]) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        if tz:
            df = df.tz_localize(tz) if df.index.tz is None else df.tz_convert(tz)
        return df
    # fall back to a time column if provided or discoverable
    if time_col is None:
        for c in ["time", "Time", "timestamp", "Timestamp", "DateTime", "datetime"]:
            if c in df.columns:
                time_col = c
                break
    if time_col is None or time_col not in df.columns:
        raise SchemaError("No DatetimeIndex and no time column found. Provide `time_col`.")
    idx = pd.to_datetime(df[time_col], errors="coerce", utc=False)
    if idx.isna().all():
        raise SchemaError(f"Could not parse datetimes from column '{time_col}'.")
    df = df.set_index(idx).drop(columns=[time_col], errors="ignore")
    if tz:
        df = df.tz_localize(tz) if df.index.tz is None else df.tz_convert(tz)
    return df


def _normalize_columns(df: pd.DataFrame, aliases: Optional[Dict[str, str]]) -> pd.DataFrame:
    # build a case-insensitive alias map
    amap = {k.lower(): v for k, v in (aliases or DEFAULT_ALIASES).items()}
    new_cols = {}
    for c in df.columns:
        key = str(c).lower().strip()
        new_cols[c] = amap.get(key, c)
    df = df.rename(columns=new_cols)
    # collapse duplicate columns created by aliasing (first non-null wins)
    if len(set(new_cols.values())) != len(new_cols.values()):
        df = df.groupby(level=0, axis=1).first()
    return df


def infer_sampling_hz(index: pd.DatetimeIndex) -> float:
    """Infer sampling frequency (Hz) from a datetime index.

    Uses the median Δt; robust to small gaps. Returns NaN if not inferable.
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 5:
        return float("nan")
    dt = np.diff(index.view("int64"))  # ns
    dt = dt[dt > 0]
    if dt.size == 0:
        return float("nan")
    med = np.median(dt)  # ns
    if med <= 0:
        return float("nan")
    hz = 1.0 / (med * 1e-9)
    # round to “nice” SR rates (10/20/32/40/etc.) if very close
    for cand in (5, 8, 10, 12.5, 16, 20, 25, 32, 40, 50):
        if abs(hz - cand) / cand < 0.01:
            return float(cand)
    return float(hz)


def _validate_schema(df: pd.DataFrame, require: Iterable[str] = ("T", "u", "v", "w")) -> None:
    missing = [c for c in require if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")


def _read_one(path: Path, *, time_col: Optional[str], tz: Optional[str], aliases: Optional[Dict[str, str]]) -> pd.DataFrame:
    if _is_parquet(path):
        df = pd.read_parquet(path)
    else:
        # Use pandas fast engine; users can pass dtype kwargs by preloading externally if needed
        df = pd.read_csv(path)
    df = _normalize_columns(df, aliases)
    df = _coerce_datetime_index(df, time_col=time_col, tz=tz)
    return df.sort_index()


def _concat_by_time(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    # Concatenate and drop exact duplicate timestamps favoring first occurrence
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, axis=0)
    out = out[~out.index.duplicated(keep="first")]
    return out.sort_index()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def read_highfreq(
    source: str | Path | Iterable[str | Path] | pd.DataFrame,
    *,
    freq_hz: Optional[float] = None,
    time_col: Optional[str] = None,
    tz: Optional[str] = None,
    aliases: Optional[Dict[str, str]] = None,
    require: Iterable[str] = ("T", "u", "v", "w"),
) -> pd.DataFrame:
    """Load high-frequency data and normalize it to the SR schema.

    Parameters
    ----------
    source : path-like | iterable of path-like | pd.DataFrame
        A CSV/Parquet file, a glob (e.g., ``"data/2024/*.parquet"``), a list of files,
        or a preloaded DataFrame.
    freq_hz : float, optional
        Sampling frequency (Hz). If ``None``, will be **inferred** from the index.
    time_col : str, optional
        Name of the timestamp column when the file does not have a DatetimeIndex.
    tz : str, optional
        Timezone name (e.g., "UTC", "America/Los_Angeles"). If provided, the index is
        localized or converted to this zone.
    aliases : dict, optional
        Column alias map to canonical names (merged with `DEFAULT_ALIASES`).
    require : iterable of str, default ("T","u","v","w")
        Columns you require to be present. Raise `SchemaError` if missing.

    Returns
    -------
    pd.DataFrame
        Datetime-indexed high-frequency DataFrame with canonical columns:
        at least ``["T","u","v","w"]``. Optional columns (if present) are left as-is,
        e.g., ``["Rn","G","P_Pa"]``.

    Notes
    -----
    - Large multi-file reads: pass a list or a glob—files will be concatenated in
      chronological order and duplicate timestamps dropped (keeping first).
    - If you need strict dtypes or custom parsing for a particular sensor export,
      pre-load with pandas and pass the DataFrame directly to `source`.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
        df = _normalize_columns(df, aliases)
        df = _coerce_datetime_index(df, time_col=time_col, tz=tz)
        df = df.sort_index()
    else:
        # Normalize to a list of paths (support glob)
        paths: list[Path] = []
        if isinstance(source, (str, Path)):
            s = str(source)
            if any(ch in s for ch in "*?[]"):
                paths = [Path(p) for p in sorted(glob.glob(s))]
            else:
                paths = [Path(s)]
        else:
            paths = [Path(p) for p in source]

        if not paths:
            raise FileNotFoundError("No input files matched.")
        dfs = [_read_one(p, time_col=time_col, tz=tz, aliases=aliases) for p in paths]
        df = _concat_by_time(dfs)

    # Validate required columns and coerce to float
    _validate_schema(df, require=require)
    for c in ["T", "u", "v", "w"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    # Attach/refresh a hint about sampling frequency
    df.attrs = dict(df.attrs)  # ensure mutable
    df.attrs["fs_hz"] = float(freq_hz) if freq_hz is not None else float(infer_sampling_hz(df.index))
    return df


def write_flux_timeseries(
    df: pd.DataFrame,
    path: str | Path,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    engine: str = "parquet",
) -> None:
    """Write block-level SR outputs to disk with minimal metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Tidy block-level results (e.g., from `pipeline.run_surface_renewal`).
    path : path-like
        Output path. Extension determines format if `engine="auto"`; otherwise
        Parquet or CSV based on `engine`.
    metadata : dict, optional
        Free-form metadata to store alongside the table (added to `df.attrs`).
    engine : {"parquet","csv"}, default "parquet"
        Output format.

    Notes
    -----
    - For Parquet, attributes are not stored natively; we embed minimal metadata
      as columns with the `__meta:` prefix for portability.
    """
    out = df.copy()
    meta = metadata or {}
    # persist a few attrs in columns (simple & portable)
    for k, v in meta.items():
        out[f"__meta:{k}"] = v

    p = Path(path)
    if engine == "csv" or p.suffix.lower() == ".csv":
        out.to_csv(p, index=True)
    else:
        out.to_parquet(p, index=True)
