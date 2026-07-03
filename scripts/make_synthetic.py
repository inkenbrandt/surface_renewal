"""Generate a synthetic high-frequency CSV for the surface-renewal pipeline.

The output has a ``time`` column (auto-discovered by
:func:`surface_renewal.io.read_highfreq`) plus the required ``T``, ``u``, ``v``,
``w`` columns and optional ``Rn``/``G`` radiation terms. The temperature series
carries a sinusoidal ramp so every SR method (snyder/chen97/fvs/castellvi/
wavelet) has structure to work with.

Usable two ways:

- Programmatically::

    from scripts.make_synthetic import make_synthetic_frame, write_synthetic_csv

- As a CLI::

    python scripts/make_synthetic.py out.csv --fs 10 --minutes 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_synthetic_frame(
    *,
    fs: float = 10.0,
    minutes: float = 30.0,
    seed: int = 0,
    with_radiation: bool = True,
) -> pd.DataFrame:
    """Return a datetime-indexed synthetic high-frequency DataFrame.

    Parameters
    ----------
    fs : float, default 10.0
        Sampling frequency (Hz).
    minutes : float, default 30.0
        Duration of the record (minutes).
    seed : int, default 0
        Seed for the random generator (reproducible output).
    with_radiation : bool, default True
        If True, include ``Rn`` and ``G`` columns so the LE residual is exercised.
    """
    rng = np.random.default_rng(seed)
    n_rows = int(round(fs * 60.0 * minutes))
    t = np.arange(n_rows) / fs

    # Temperature: sinusoidal ramp (amplitude 0.3 K, 60 s period) plus noise, so
    # the structure-function / wavelet ramp detectors have a signal to lock onto.
    T = 298.15 + 0.3 * np.sin(2 * np.pi * t / 60.0) + rng.standard_normal(n_rows) * 0.02

    # Correlated wind components with a mean flow (keeps rotation/u* well-posed).
    u = 2.5 + rng.standard_normal(n_rows) * 0.5 + 0.5 * np.sin(2 * np.pi * t / 60.0)
    v = 0.1 + 0.5 * (u - 2.5) + rng.standard_normal(n_rows) * 0.3
    w = 0.2 * (u - 2.5) + rng.standard_normal(n_rows) * 0.15

    start = pd.Timestamp("2023-01-01 00:00:00")
    index = start + pd.to_timedelta(t, unit="s")

    data = {"T": T, "u": u, "v": v, "w": w}
    if with_radiation:
        data["Rn"] = 500 + rng.standard_normal(n_rows) * 50
        data["G"] = 50 + rng.standard_normal(n_rows) * 10
    return pd.DataFrame(data, index=index)


def write_synthetic_csv(path: str | Path, **kwargs) -> Path:
    """Write a synthetic frame to ``path`` as CSV (with a ``time`` column)."""
    df = make_synthetic_frame(**kwargs)
    p = Path(path)
    df.to_csv(p, index_label="time")
    return p


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="make_synthetic",
        description="Write a synthetic high-frequency CSV for the SR pipeline.",
    )
    p.add_argument("output", help="Output CSV path.")
    p.add_argument("--fs", type=float, default=10.0, help="Sampling frequency (Hz).")
    p.add_argument("--minutes", type=float, default=30.0, help="Record length (minutes).")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--no-radiation", action="store_true",
                   help="Omit the Rn/G columns.")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = _build_argparser().parse_args(argv)
    out = write_synthetic_csv(
        ns.output,
        fs=ns.fs,
        minutes=ns.minutes,
        seed=ns.seed,
        with_radiation=not ns.no_radiation,
    )
    print(f"Wrote synthetic CSV: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
