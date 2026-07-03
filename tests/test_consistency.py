"""Consistency sweep tests.

1. The empty-result column list built by ``run_surface_renewal`` must match the
   keys actually returned by ``_compute_block_flux`` for *every* method branch.
2. The CLI (`python -m surface_renewal.compute`) must exit 0 for every method on
   a generated synthetic CSV.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from surface_renewal.pipeline import (
    PipelineConfig,
    run_surface_renewal,
    _ensure_df,
    _preprocess_df,
    _compute_block_flux,
)
from scripts.make_synthetic import make_synthetic_frame, write_synthetic_csv

ALL_METHODS = ("snyder", "chen97", "fvs", "castellvi", "wavelet")
Z_M = 2.0  # height above zero-plane displacement, needed by fvs/castellvi


def _cfg(method: str) -> PipelineConfig:
    # rotation="double" keeps the small synthetic block well-posed; z_m satisfies
    # the height-dependent methods.
    return PipelineConfig(fs=10, method=method, rotation="double", z_m=Z_M)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_empty_cols_match_block_keys(method):
    """Empty-frame columns == keys returned by _compute_block_flux (per method)."""
    cfg = _cfg(method)

    # Columns produced on the empty-data path.
    empty_df = make_synthetic_frame(fs=10, minutes=30).iloc[0:0]
    empty_cols = set(run_surface_renewal(empty_df, cfg=cfg).columns)

    # Keys produced by the real per-block computation for this method branch.
    df = make_synthetic_frame(fs=10, minutes=30, seed=1)
    df_prep = _preprocess_df(_ensure_df(df, fs=cfg.fs), cfg)
    block_keys = set(_compute_block_flux(df_prep, cfg).keys())

    assert block_keys == empty_cols, (
        f"[{method}] mismatch between empty-frame columns and block keys:\n"
        f"  only in empty cols: {sorted(empty_cols - block_keys)}\n"
        f"  only in block keys: {sorted(block_keys - empty_cols)}"
    )


@pytest.fixture(scope="module")
def synthetic_csv(tmp_path_factory) -> Path:
    """A synthetic high-frequency CSV shared across the CLI method runs."""
    path = tmp_path_factory.mktemp("synthetic") / "highfreq.csv"
    return write_synthetic_csv(path, fs=10, minutes=30, seed=2)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_cli_runs_per_method(method, synthetic_csv, tmp_path):
    """`python -m surface_renewal.compute --method M ...` exits 0 for every M."""
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / f"out_{method}.csv"
    cmd = [
        sys.executable, "-m", "surface_renewal.compute",
        str(synthetic_csv),
        "--fs", "10",
        "--method", method,
        "--rotation", "double",
        "--z-m", str(Z_M),
        "--out", str(out_path),
    ]
    # The subprocess does not inherit pytest's `pythonpath`, so put `src` on
    # PYTHONPATH explicitly to make the package importable regardless of install.
    env = dict(os.environ)
    src_dir = str(repo_root / "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(repo_root), env=env,
    )
    assert proc.returncode == 0, (
        f"CLI failed for method={method} (rc={proc.returncode})\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert out_path.exists()
