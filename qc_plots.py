#!/usr/bin/env python3
"""
qc_plots.py

QC visualization for glacier-scene outputs.

Creates:
1. Per-scene QC panels with:
   - RGB image
   - NDSI heatmap
   - glacier mask overlay on RGB

2. Per-glacier time-series plots of area through time

3. Per-glacier thumbnail strips showing visual change over time

Expected inputs:
- data/results/qc_scene_table.csv (preferred)
  or
- data/results/glacier_area_timeseries.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

import config as cfg


def sanitize_id(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(s))


def ensure_dirs() -> None:
    cfg.QC_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.QC_SCENES_DIR.mkdir(parents=True, exist_ok=True)
    cfg.QC_TIMESERIES_DIR.mkdir(parents=True, exist_ok=True)
    cfg.QC_STRIPS_DIR.mkdir(parents=True, exist_ok=True)


def load_qc_table() -> pd.DataFrame:
    qc_path = cfg.RESULTS_DIR / "qc_scene_table.csv"
    ts_path = cfg.RESULTS_DIR / "glacier_area_timeseries.csv"

    if qc_path.exists():
        df = pd.read_csv(qc_path)
    elif ts_path.exists():
        df = pd.read_csv(ts_path)
    else:
        raise FileNotFoundError(f"Could not find either {qc_path} or {ts_path}")

    if df.empty:
        print("QC input table is empty.")
        return df

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    sort_cols = [c for c in ["glacier_id", "date", "item_id"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def load_single_band(path: str) -> tuple[np.ndarray, float | None]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
    return arr, nodata


def load_rgb(path: str) -> tuple[np.ndarray, float | None]:
    with rasterio.open(path) as src:
        arr = src.read([1, 2, 3]).astype(np.float32)
        nodata = src.nodata
    arr = np.moveaxis(arr, 0, -1)
    return arr, nodata


def normalize_rgb(rgb: np.ndarray, nodata: float | None = None) -> np.ndarray:
    rgb_disp = rgb.copy()

    if nodata is not None:
        rgb_disp[rgb_disp == nodata] = np.nan

    finite = np.isfinite(rgb_disp)
    if not np.any(finite):
        return np.zeros_like(rgb_disp, dtype=np.float32)

    vals = rgb_disp[finite]
    lo = np.nanpercentile(vals, cfg.RGB_STRETCH_LOW)
    hi = np.nanpercentile(vals, cfg.RGB_STRETCH_HIGH)

    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1e-6

    rgb_disp = (rgb_disp - lo) / (hi - lo)
    rgb_disp = np.clip(rgb_disp, 0, 1)
    rgb_disp = np.nan_to_num(rgb_disp, nan=0.0)

    return rgb_disp


def build_glacier_mask(ndsi: np.ndarray, nodata: float | None, threshold: float) -> np.ndarray:
    finite = np.isfinite(ndsi)
    if nodata is not None:
        finite &= (ndsi != nodata)
    return finite & (ndsi >= threshold)


def overlay_mask_on_rgb(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    out = rgb.copy()
    if out.ndim != 3 or out.shape[2] != 3:
        raise ValueError("RGB array must have shape (rows, cols, 3).")

    red = np.zeros_like(out)
    red[..., 0] = 1.0

    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * red[m]
    return np.clip(out, 0, 1)


def make_scene_panel(row: pd.Series) -> Optional[Path]:
    if "rgb_path" not in row or not isinstance(row["rgb_path"], str) or not row["rgb_path"]:
        print(f"Skipping scene without rgb_path: {row.get('item_id', 'unknown')}")
        return None

    rgb_path = Path(row["rgb_path"])
    ndsi_path = Path(row["ndsi_path"])

    if not rgb_path.exists():
        print(f"Missing RGB file: {rgb_path}")
        return None
    if not ndsi_path.exists():
        print(f"Missing NDSI file: {ndsi_path}")
        return None

    rgb, rgb_nodata = load_rgb(str(rgb_path))
    ndsi, ndsi_nodata = load_single_band(str(ndsi_path))

    rgb_disp = normalize_rgb(rgb, rgb_nodata)
    glacier_mask = build_glacier_mask(ndsi, ndsi_nodata, cfg.NDSI_THRESHOLD)
    rgb_overlay = overlay_mask_on_rgb(rgb_disp, glacier_mask, alpha=cfg.GLACIER_OVERLAY_ALPHA)

    ndsi_plot = ndsi.copy()
    if ndsi_nodata is not None:
        ndsi_plot[ndsi_plot == ndsi_nodata] = np.nan

    glacier_id = row["glacier_id"]
    item_id = row["item_id"]
    date = row["date"]
    date_str = "unknown_date" if pd.isna(date) else pd.Timestamp(date).strftime("%Y-%m-%d")

    area_str = ""
    if "area_km2" in row and pd.notna(row["area_km2"]):
        area_str = f" | area={row['area_km2']:.3f} km²"

    vf_str = ""
    if "valid_fraction" in row and pd.notna(row["valid_fraction"]):
        vf_str = f" | valid_fraction={row['valid_fraction']:.3f}"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb_disp)
    axes[0].set_title("RGB")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    im1 = axes[1].imshow(ndsi_plot, vmin=-0.5, vmax=1.0)
    axes[1].set_title("NDSI")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(rgb_overlay)
    axes[2].set_title(f"RGB + NDSI ≥ {cfg.NDSI_THRESHOLD}")
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    fig.suptitle(f"{glacier_id} | {date_str} | {item_id}{area_str}{vf_str}", fontsize=10)
    fig.tight_layout()

    glacier_dir = cfg.QC_SCENES_DIR / sanitize_id(glacier_id)
    glacier_dir.mkdir(parents=True, exist_ok=True)

    out_path = glacier_dir / f"{date_str}_{sanitize_id(item_id)}_qc.png"
    fig.savefig(out_path, dpi=cfg.QC_DPI, bbox_inches="tight")
    plt.close(fig)

    return out_path


def make_scene_panels(df: pd.DataFrame) -> None:
    use_df = df.copy()

    if cfg.QC_MAX_SCENES is not None:
        use_df = use_df.head(cfg.QC_MAX_SCENES)

    print(f"Making scene QC panels for {len(use_df)} scenes...")
    for _, row in use_df.iterrows():
        try:
            out = make_scene_panel(row)
            if out is not None:
                print(f"Wrote {out}")
        except Exception as e:
            print(f"Failed scene QC for {row.get('item_id', 'unknown')}: {e}")


def make_timeseries_all(df: pd.DataFrame) -> None:
    if df.empty or "area_km2" not in df.columns:
        return

    plt.figure(figsize=(10, 6))
    for glacier_id, sub in df.groupby("glacier_id"):
        sub = sub.sort_values("date")
        plt.plot(sub["date"], sub["area_km2"], marker="o", label=glacier_id)

    plt.xlabel("Date")
    plt.ylabel("Area (km²)")
    plt.title("Glacier area through time")
    plt.legend()
    plt.tight_layout()

    out_path = cfg.QC_TIMESERIES_DIR / "glacier_area_timeseries_all.png"
    plt.savefig(out_path, dpi=cfg.QC_DPI)
    plt.close()
    print(f"Wrote {out_path}")


def make_timeseries_per_glacier(df: pd.DataFrame) -> None:
    if df.empty or "area_km2" not in df.columns:
        return

    for glacier_id, sub in df.groupby("glacier_id"):
        sub = sub.sort_values("date")

        plt.figure(figsize=(9, 5))
        plt.plot(sub["date"], sub["area_km2"], marker="o")
        plt.xlabel("Date")
        plt.ylabel("Area (km²)")
        plt.title(f"Glacier area through time: {glacier_id}")
        plt.tight_layout()

        out_path = cfg.QC_TIMESERIES_DIR / f"glacier_area_timeseries_{sanitize_id(glacier_id)}.png"
        plt.savefig(out_path, dpi=cfg.QC_DPI)
        plt.close()
        print(f"Wrote {out_path}")


def make_glacier_strip(glacier_id: str, sub: pd.DataFrame) -> Optional[Path]:
    sub = sub.sort_values("date").copy()

    if cfg.QC_STRIP_MAX_SCENES is not None and len(sub) > cfg.QC_STRIP_MAX_SCENES:
        idx = np.linspace(0, len(sub) - 1, cfg.QC_STRIP_MAX_SCENES).round().astype(int)
        sub = sub.iloc[idx].copy()

    n = len(sub)
    if n == 0:
        return None

    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.8))
    if n == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, sub.iterrows()):
        rgb_path = row.get("rgb_path", "")
        ndsi_path = row.get("ndsi_path", "")

        if not isinstance(rgb_path, str) or not rgb_path or not Path(rgb_path).exists():
            ax.axis("off")
            ax.set_title("missing RGB")
            continue
        if not isinstance(ndsi_path, str) or not ndsi_path or not Path(ndsi_path).exists():
            ax.axis("off")
            ax.set_title("missing NDSI")
            continue

        rgb, rgb_nodata = load_rgb(rgb_path)
        ndsi, ndsi_nodata = load_single_band(ndsi_path)

        rgb_disp = normalize_rgb(rgb, rgb_nodata)
        glacier_mask = build_glacier_mask(ndsi, ndsi_nodata, cfg.NDSI_THRESHOLD)
        rgb_overlay = overlay_mask_on_rgb(rgb_disp, glacier_mask, alpha=cfg.GLACIER_OVERLAY_ALPHA)

        ax.imshow(rgb_overlay)
        ax.set_xticks([])
        ax.set_yticks([])

        date_str = "unknown" if pd.isna(row["date"]) else pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
        area_str = f"{row['area_km2']:.2f} km²" if "area_km2" in row and pd.notna(row["area_km2"]) else ""
        ax.set_title(f"{date_str}\n{area_str}", fontsize=8)

    fig.suptitle(f"{glacier_id}: RGB + glacier mask through time", fontsize=12)
    fig.tight_layout()

    out_path = cfg.QC_STRIPS_DIR / f"glacier_strip_{sanitize_id(glacier_id)}.png"
    fig.savefig(out_path, dpi=cfg.QC_DPI, bbox_inches="tight")
    plt.close(fig)

    return out_path


def make_all_glacier_strips(df: pd.DataFrame) -> None:
    if df.empty:
        return

    for glacier_id, sub in df.groupby("glacier_id"):
        try:
            out = make_glacier_strip(glacier_id, sub)
            if out is not None:
                print(f"Wrote {out}")
        except Exception as e:
            print(f"Failed strip for {glacier_id}: {e}")


def main() -> None:
    ensure_dirs()
    df = load_qc_table()

    if df.empty:
        return

    if cfg.MAKE_SCENE_QC_PANELS:
        make_scene_panels(df)

    if cfg.MAKE_QC_TIMESERIES:
        make_timeseries_all(df)
        make_timeseries_per_glacier(df)

    if cfg.MAKE_QC_STRIPS:
        make_all_glacier_strips(df)


if __name__ == "__main__":
    main()