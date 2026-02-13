#!/usr/bin/env python3
"""
build_cache.py

Heavy step (run only when preprocessing choices change):
- Query Planetary Computer STAC for Landsat Collection 2 Level-2 scenes intersecting ROI
- Stream only ROI pixels (COG window reads) for GREEN, SWIR1, QA_PIXEL
- Mask clouds/shadows via QA_PIXEL bits
- Compute per-scene NDSI
- Write per-pixel median NDSI composite for each year to GeoTIFF cache

Outputs (per year):
  data/cache/ndsi_median_YYYY.tif
  data/cache/ndsi_validcount_YYYY.tif   (optional QC)
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import transform_bounds
from pystac_client import Client
import planetary_computer as pc

import config as cfg


# ----------------------------
# Helpers
# ----------------------------
def load_roi(roi_path: str) -> gpd.GeoDataFrame:
    """Load ROI file, dissolve into a single geometry, ensure EPSG:4326."""
    gdf = gpd.read_file(roi_path)
    if gdf.empty:
        raise ValueError(f"ROI file has no features: {roi_path}")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    # Fix minor invalid geometries and dissolve to one shape
    gdf["geometry"] = gdf.geometry.buffer(0)
    geom = gdf.unary_union
    return gpd.GeoDataFrame({"geometry": [geom]}, crs="EPSG:4326")


def stac_search_items(catalog: Client, roi_geom, year: int) -> List:
    """Search Landsat scenes for a given year+season window, intersecting ROI."""
    dt = f"{year}-{cfg.START_MM_DD}/{year}-{cfg.END_MM_DD}"

    # Prefer filtering by scene-level cloud cover metadata if supported.
    try:
        search = catalog.search(
            collections=[cfg.COLLECTION],
            intersects=roi_geom,
            datetime=dt,
            query={"eo:cloud_cover": {"lt": cfg.MAX_CLOUD}},
            max_items=500,
        )
        items = list(search.get_items())
        if items:
            return items
    except Exception:
        pass

    # Fallback: no cloud-cover query (still cloud-mask per pixel later)
    search = catalog.search(
        collections=[cfg.COLLECTION],
        intersects=roi_geom,
        datetime=dt,
        max_items=500,
    )
    return list(search.get_items())


def asset_has_common_name(asset, common_name: str) -> bool:
    """Return True if this asset advertises eo:bands with the given common_name."""
    bands = asset.extra_fields.get("eo:bands") or []
    return any(b.get("common_name") == common_name for b in bands)


def pick_band_asset(item, common_names: Iterable[str]) -> str:
    """
    Choose an asset key by searching for eo:bands common_name values.
    This avoids hard-coding Landsat band IDs across sensors.
    """
    for cn in common_names:
        for key, asset in item.assets.items():
            if asset_has_common_name(asset, cn):
                return key
    raise KeyError(f"No asset found with common_name in {list(common_names)} for item {item.id}")


def read_window_as_array(asset_href: str, roi_bounds_wgs84, out_dtype=np.float32):
    """
    Stream-read only the window covering roi_bounds (WGS84) from a COG asset.
    Returns: (array, profile)
    """
    with rasterio.open(asset_href) as src:
        # Convert ROI bounds to the asset CRS, then compute a read window.
        bounds_src = transform_bounds("EPSG:4326", src.crs, *roi_bounds_wgs84, densify_pts=21)
        window = src.window(*bounds_src).round_offsets().round_lengths()

        arr = src.read(1, window=window, boundless=True, fill_value=src.nodata).astype(out_dtype, copy=False)

        profile = src.profile.copy()
        profile.update(
            height=arr.shape[0],
            width=arr.shape[1],
            transform=src.window_transform(window),
        )
        return arr, profile


def qa_good_mask(qa_pixel: np.ndarray) -> np.ndarray:
    """True where QA_PIXEL indicates usable (non-cloud/shadow/fill/cirrus) pixels."""
    bad = cfg.BAD_BITS | (cfg.WATER if cfg.MASK_WATER else 0)
    return (qa_pixel.astype(np.uint16) & bad) == 0


def scale_sr(dn: np.ndarray) -> np.ndarray:
    """Apply USGS Landsat C2 L2 surface reflectance scaling."""
    return dn * cfg.SR_SCALE + cfg.SR_OFFSET


def maybe_debug_subset(items: List) -> List:
    """Optional speed-up: sort by cloud cover and keep only first N scenes."""
    if cfg.DEBUG_SORT_BY_CLOUD:
        items = sorted(items, key=lambda it: it.properties.get("eo:cloud_cover", 999))
    if cfg.DEBUG:
        items = items[: cfg.DEBUG_MAX_SCENES]
    return items


# ----------------------------
# Main yearly builder
# ----------------------------
def build_year(catalog: Client, roi_gdf: gpd.GeoDataFrame, year: int) -> None:
    cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    roi_geom = roi_gdf.geometry.iloc[0].__geo_interface__   # for STAC intersects
    roi_bounds = tuple(roi_gdf.total_bounds)               # (minx, miny, maxx, maxy) in EPSG:4326

    items = stac_search_items(catalog, roi_geom, year)
    if not items:
        print(f"[{year}] No scenes found for ROI/time window.")
        return

    items = maybe_debug_subset(items)

    # Sign items so asset hrefs are readable (adds a short-lived token if required).
    items = [pc.sign(it) for it in items]

    ndsi_stack = []
    used = 0
    profile = None

    for it in items:
        try:
            green_key = pick_band_asset(it, common_names=["green"])
            swir1_key = pick_band_asset(it, common_names=["swir16", "swir1"])
            qa_key = "QA_PIXEL"

            green_href = it.assets[green_key].href
            swir1_href = it.assets[swir1_key].href
            qa_href = it.assets[qa_key].href

            green_dn, profile = read_window_as_array(green_href, roi_bounds, out_dtype=np.float32)
            swir1_dn, _ = read_window_as_array(swir1_href, roi_bounds, out_dtype=np.float32)
            qa_pix, _ = read_window_as_array(qa_href, roi_bounds, out_dtype=np.uint16)

            good = qa_good_mask(qa_pix)

            green = scale_sr(green_dn)
            swir1 = scale_sr(swir1_dn)

            denom = green + swir1
            ndsi = np.full_like(green, np.nan, dtype=np.float32)

            valid = good & np.isfinite(denom) & (np.abs(denom) > 1e-6)
            ndsi[valid] = (green[valid] - swir1[valid]) / denom[valid]

            ndsi_stack.append(ndsi)
            used += 1

        except Exception as e:
            print(f"[{year}] Skip item {it.id}: {e}")

    if used == 0 or profile is None:
        print(f"[{year}] No usable scenes after filtering.")
        return

    stack = np.stack(ndsi_stack, axis=0)  # (n_scenes, rows, cols)
    ndsi_med = np.nanmedian(stack, axis=0).astype(np.float32)

    out = np.where(np.isfinite(ndsi_med), ndsi_med, cfg.NODATA_FLOAT).astype(np.float32)

    out_profile = profile.copy()
    out_profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        compress=cfg.GTIFF_COMPRESS,
        nodata=cfg.NODATA_FLOAT,
        tiled=True,
    )

    out_path = cfg.CACHE_DIR / f"ndsi_median_{year}.tif"
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(out, 1)

    if cfg.WRITE_VALIDCOUNT:
        validcount = np.sum(np.isfinite(stack), axis=0).astype(np.uint16)

        vc_profile = profile.copy()
        vc_profile.update(
            driver="GTiff",
            dtype="uint16",
            count=1,
            compress=cfg.GTIFF_COMPRESS,
            nodata=0,
            tiled=True,
        )
        vc_path = cfg.CACHE_DIR / f"ndsi_validcount_{year}.tif"
        with rasterio.open(vc_path, "w", **vc_profile) as dst:
            dst.write(validcount, 1)

    msg = f"[{year}] Wrote {out_path.name} using {used} scenes"
    if cfg.DEBUG:
        msg += f" (DEBUG: max {cfg.DEBUG_MAX_SCENES})"
    print(msg + ".")


def main() -> None:
    roi = load_roi(cfg.ROI_PATH)
    catalog = Client.open(cfg.STAC_URL)

    for year in range(cfg.START_YEAR, cfg.END_YEAR + 1):
        build_year(catalog, roi, year)


if __name__ == "__main__":
    main()
