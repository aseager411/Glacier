#!/usr/bin/env python3
"""
analyze.py

Read accepted cached scenes from build_cache.py and compute glacier area
for each accepted timestamp.

This script:
- reads the master scene index from build_cache.py
- loads each cached NDSI raster
- thresholds NDSI to define glacier pixels
- computes glacier area from raster pixel area
- writes one master results CSV for all glaciers and timestamps
- writes one per-glacier CSV for convenience
- optionally writes a lightweight annual summary CSV

Intended separation of responsibilities:
- build_cache.py : scene discovery, masking, caching
- analyze.py     : area extraction, tabular outputs
- qc_plots.py    : visual inspection / overlays
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import rasterio

import config as cfg


# ---------------------------------------------------
# I/O helpers
# ---------------------------------------------------
def load_scene_index() -> pd.DataFrame:
    """
    Load the accepted-scene index produced by build_cache.py.
    """
    path = cfg.SCENE_INDEX_PATH
    if not path.exists():
        raise FileNotFoundError(f"Scene index not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        print("Scene index is empty.")
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def ensure_results_dirs() -> None:
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.RESULTS_PER_GLACIER_DIR.mkdir(parents=True, exist_ok=True)
    cfg.RESULTS_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_id(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(s))


# ---------------------------------------------------
# Raster / area helpers
# ---------------------------------------------------
def compute_pixel_area_m2(transform) -> float:
    """
    Compute pixel area from affine transform.
    Assumes north-up raster with square/rectangular pixels.
    """
    return abs(transform.a * transform.e)


def load_ndsi_raster(ndsi_path: str) -> tuple[np.ndarray, float | None, object]:
    """
    Load cached NDSI raster and return (array, nodata, transform).
    """
    with rasterio.open(ndsi_path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        transform = src.transform

    return arr, nodata, transform


def glacier_mask_from_ndsi(arr: np.ndarray, nodata: float | None, threshold: float) -> np.ndarray:
    """
    Build a glacier mask from NDSI threshold.
    """
    finite = np.isfinite(arr)
    if nodata is not None:
        finite &= (arr != nodata)

    mask = finite & (arr >= threshold)
    return mask


def compute_scene_metrics(ndsi_path: str, threshold: float) -> Dict:
    """
    Compute glacier-area metrics for one cached NDSI raster.
    """
    arr, nodata, transform = load_ndsi_raster(ndsi_path)
    glacier_mask = glacier_mask_from_ndsi(arr, nodata, threshold)

    n_glacier_pixels = int(np.sum(glacier_mask))
    pixel_area_m2 = float(compute_pixel_area_m2(transform))
    area_m2 = float(n_glacier_pixels * pixel_area_m2)
    area_km2 = float(area_m2 / 1e6)

    return {
        "n_glacier_pixels": n_glacier_pixels,
        "pixel_area_m2": pixel_area_m2,
        "area_m2": area_m2,
        "area_km2": area_km2,
    }


# ---------------------------------------------------
# Core analysis
# ---------------------------------------------------
def analyze_all_scenes(scene_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute glacier area for every accepted scene in the scene index.
    """
    rows: List[Dict] = []

    for _, row in scene_df.iterrows():
        ndsi_path = row["ndsi_path"]

        try:
            metrics = compute_scene_metrics(
                ndsi_path=ndsi_path,
                threshold=cfg.NDSI_THRESHOLD,
            )

            out = {
                "glacier_id": row["glacier_id"],
                "year": int(row["year"]),
                "date": row["date"],
                "item_id": row["item_id"],
                "cloud_cover": row.get("cloud_cover", np.nan),
                "valid_fraction": row.get("valid_fraction", np.nan),
                "n_valid": row.get("n_valid", np.nan),
                "n_roi": row.get("n_roi", np.nan),
                "ndsi_path": row["ndsi_path"],
                "validmask_path": row.get("validmask_path", ""),
                "rgb_path": row.get("rgb_path", ""),
                "ndsi_threshold": cfg.NDSI_THRESHOLD,
                **metrics,
            }
            rows.append(out)

            print(
                f"[OK] {row['glacier_id']} | {str(row['date'])[:10]} | "
                f"pixels={metrics['n_glacier_pixels']} | "
                f"area_km2={metrics['area_km2']:.4f}"
            )

        except Exception as e:
            print(f"[FAIL] {ndsi_path}: {e}")

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["glacier_id", "date", "item_id"]).reset_index(drop=True)

    return result


# ---------------------------------------------------
# Output writers
# ---------------------------------------------------
def write_master_results(results_df: pd.DataFrame) -> Path:
    out_csv = cfg.RESULTS_DIR / "glacier_area_timeseries.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    return out_csv


def write_per_glacier_results(results_df: pd.DataFrame) -> None:
    """
    Convenience outputs. Not the main data product.
    """
    for glacier_id, sub in results_df.groupby("glacier_id"):
        safe_id = sanitize_id(glacier_id)
        out_csv = cfg.RESULTS_PER_GLACIER_DIR / f"glacier_area_timeseries_{safe_id}.csv"
        sub.sort_values("date").to_csv(out_csv, index=False)
        print(f"Wrote {out_csv}")


def write_annual_summary(results_df: pd.DataFrame) -> Path:
    """
    Write a simple annual summary for convenience.

    Since there may be multiple accepted scenes per year, this keeps:
    - n_scenes
    - min area
    - median area
    - max area
    """
    grouped = (
        results_df.groupby(["glacier_id", "year"], dropna=False)
        .agg(
            n_scenes=("area_km2", "count"),
            area_km2_min=("area_km2", "min"),
            area_km2_median=("area_km2", "median"),
            area_km2_max=("area_km2", "max"),
            valid_fraction_median=("valid_fraction", "median"),
        )
        .reset_index()
        .sort_values(["glacier_id", "year"])
    )

    out_csv = cfg.RESULTS_SUMMARY_DIR / "glacier_area_annual_summary.csv"
    grouped.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    return out_csv


def write_qc_scene_table(results_df: pd.DataFrame) -> Path:
    """
    Write a QC-oriented table for later use by qc_plots.py.

    This keeps all scene-level metadata and file paths in one place.
    """
    qc_cols = [
        "glacier_id",
        "year",
        "date",
        "item_id",
        "cloud_cover",
        "valid_fraction",
        "n_valid",
        "n_roi",
        "n_glacier_pixels",
        "area_m2",
        "area_km2",
        "ndsi_threshold",
        "ndsi_path",
        "validmask_path",
        "rgb_path",
    ]

    qc_df = results_df[qc_cols].copy()
    out_csv = cfg.RESULTS_DIR / "qc_scene_table.csv"
    qc_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    return out_csv


# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main() -> None:
    ensure_results_dirs()

    scene_df = load_scene_index()
    if scene_df.empty:
        print("No accepted scenes found in scene_index.csv.")
        return

    results_df = analyze_all_scenes(scene_df)
    if results_df.empty:
        print("No scene metrics were computed.")
        return

    write_master_results(results_df)
    write_per_glacier_results(results_df)

    if cfg.WRITE_ANNUAL_SUMMARY:
        write_annual_summary(results_df)

    if cfg.WRITE_QC_SCENE_TABLE:
        write_qc_scene_table(results_df)


if __name__ == "__main__":
    main()