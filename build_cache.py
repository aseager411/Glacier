#!/usr/bin/env python3
"""
build_cache.py

Cache the 3 least-cloudy Sentinel-2 late-summer scenes for each glacier polygon
using STAC metadata ranking, then generate the same cached raster outputs as before.

Workflow:
- Read glacier polygons from ROI file
- For each glacier and each year:
    - Query Planetary Computer STAC for Sentinel-2 L2A scenes intersecting glacier
    - Sort candidate scenes by eo:cloud_cover metadata
    - Keep only the least-cloudy 3 scenes for that glacier-year
    - For each selected scene:
        - Read GREEN, SWIR1, RED, BLUE, and SCL over glacier window
        - Build a glacier-polygon mask on the raster grid
        - Mask cloud/shadow/bad pixels using SCL classes
        - Compute NDSI
        - Write one NDSI GeoTIFF, one valid-mask GeoTIFF, and one RGB GeoTIFF
- Write a scene index CSV with metadata for all accepted scenes

Outputs:
  data/cache/<glacier_id>/<year>/ndsi_<date>_<itemid>.tif
  data/cache/<glacier_id>/<year>/validmask_<date>_<itemid>.tif
  data/cache/<glacier_id>/<year>/rgb_<date>_<itemid>.tif
  data/cache/scene_index.csv
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

import config as cfg


# ----------------------------
# Constants
# ----------------------------
TOP_K_SCENES_PER_YEAR = 3


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class SceneScore:
    glacier_id: str
    year: int
    item_id: str
    date: str
    cloud_cover: float
    valid_fraction: float
    n_valid: int
    n_roi: int
    ndsi_path: str = ""
    validmask_path: str = ""
    rgb_path: str = ""


# ----------------------------
# Helpers
# ----------------------------
def load_rois(roi_path: str) -> gpd.GeoDataFrame:
    """
    Load ROI file, repair minor invalid geometry, ensure EPSG:4326.
    Keeps polygons separate. Creates glacier_id if none exists.
    """
    gdf = gpd.read_file(roi_path)
    if gdf.empty:
        raise ValueError(f"ROI file has no features: {roi_path}")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()

    for candidate in ["glacier_id", "name", "Name", "id", "ID"]:
        if candidate in gdf.columns:
            gdf["glacier_id"] = gdf[candidate].astype(str)
            break
    else:
        gdf["glacier_id"] = [f"glacier_{i+1}" for i in range(len(gdf))]

    return gdf[["glacier_id", "geometry"]].copy()


def sanitize_id(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(s))


def safe_cloud_cover(item) -> float:
    """
    Cloud-cover sort key. Missing/invalid values sort last.
    """
    value = item.properties.get("eo:cloud_cover", np.inf)
    try:
        value = float(value)
    except Exception:
        value = np.inf

    if not np.isfinite(value):
        return np.inf
    return value


def safe_item_datetime_str(item) -> str:
    if item.datetime:
        return item.datetime.isoformat()
    return "unknown"


def stac_search_items(catalog: Client, roi_geom, year: int) -> List:
    """
    Search Sentinel-2 L2A scenes for a given year+season window.
    Prefer cloud-filtered results, but fall back to unfiltered search.
    """
    dt = f"{year}-{cfg.START_MM_DD}/{year}-{cfg.END_MM_DD}"

    try:
        search = catalog.search(
            collections=[cfg.COLLECTION],
            intersects=roi_geom,
            datetime=dt,
            query={"eo:cloud_cover": {"lt": cfg.MAX_CLOUD}},
            max_items=500,
        )
        items = list(search.items())
        if items:
            return items
    except Exception:
        pass

    search = catalog.search(
        collections=[cfg.COLLECTION],
        intersects=roi_geom,
        datetime=dt,
        max_items=500,
    )
    return list(search.items())


def maybe_debug_subset(items: List) -> List:
    """
    Preserve debug behavior, but sort by cloud if requested.
    """
    if cfg.DEBUG_SORT_BY_CLOUD:
        items = sorted(
            items,
            key=lambda it: (safe_cloud_cover(it), safe_item_datetime_str(it), it.id),
        )
    if cfg.DEBUG:
        items = items[: cfg.DEBUG_MAX_SCENES]
    return items


def select_top_items_by_cloud(items: List, top_k: int = TOP_K_SCENES_PER_YEAR) -> List:
    """
    Select the least-cloudy scenes using STAC metadata only.
    Ties break by date string then item id for stable ordering.
    """
    items_sorted = sorted(
        items,
        key=lambda it: (safe_cloud_cover(it), safe_item_datetime_str(it), it.id),
    )
    return items_sorted[:top_k]


def asset_has_common_name(asset, common_name: str) -> bool:
    bands = asset.extra_fields.get("eo:bands") or []
    return any(b.get("common_name") == common_name for b in bands)


def pick_band_asset(item, common_names: Iterable[str]) -> str:
    """
    Choose an asset key by eo:bands common_name.
    """
    for cn in common_names:
        for key, asset in item.assets.items():
            if asset_has_common_name(asset, cn):
                return key
    raise KeyError(f"No asset found with common_name in {list(common_names)} for item {item.id}")


def read_window_native(asset_href: str, roi_bounds_wgs84, out_dtype=np.float32):
    """
    Read ROI window in the asset's native grid.
    Returns (array, profile).
    """
    with rasterio.open(asset_href) as src:
        bounds_src = transform_bounds("EPSG:4326", src.crs, *roi_bounds_wgs84, densify_pts=21)
        window = src.window(*bounds_src).round_offsets().round_lengths()

        arr = src.read(
            1,
            window=window,
            boundless=True,
            fill_value=src.nodata,
        ).astype(out_dtype, copy=False)

        profile = src.profile.copy()
        profile.update(
            height=arr.shape[0],
            width=arr.shape[1],
            transform=src.window_transform(window),
            crs=src.crs,
        )
        return arr, profile


def read_window_match_grid(
    asset_href: str,
    ref_profile: dict,
    out_dtype=np.float32,
    resampling=Resampling.bilinear,
):
    """
    Read an asset reprojected/resampled onto the reference grid.
    """
    with rasterio.open(asset_href) as src:
        with WarpedVRT(
            src,
            crs=ref_profile["crs"],
            transform=ref_profile["transform"],
            width=ref_profile["width"],
            height=ref_profile["height"],
            resampling=resampling,
        ) as vrt:
            arr = vrt.read(1).astype(out_dtype, copy=False)

    return arr


def project_geometry_to_crs(geom_wgs84, dst_crs):
    """
    Reproject a single WGS84 geometry to destination CRS.
    """
    return gpd.GeoSeries([geom_wgs84], crs="EPSG:4326").to_crs(dst_crs).iloc[0]


def make_polygon_mask_from_projected_geom(geom_proj, out_profile) -> np.ndarray:
    """
    True inside glacier polygon, False outside, on raster grid.
    """
    mask = geometry_mask(
        [geom_proj.__geo_interface__],
        out_shape=(out_profile["height"], out_profile["width"]),
        transform=out_profile["transform"],
        invert=True,
    )
    return mask


def sentinel_screen_mask(scl: np.ndarray) -> np.ndarray:
    """
    Conservative mask used for valid-mask output / score summary.
    """
    bad_classes = {0, 1, 3, 8, 9, 10}

    if getattr(cfg, "MASK_WATER", False):
        bad_classes.add(6)

    if getattr(cfg, "MASK_SNOW_ICE", False):
        bad_classes.add(11)

    good = np.ones_like(scl, dtype=bool)
    for cls in bad_classes:
        good &= (scl != cls)
    return good


def sentinel_compute_mask(scl: np.ndarray) -> np.ndarray:
    """
    Minimal mask used for NDSI computation itself.
    Usually only exclude truly unusable pixels.
    """
    bad_classes = {0, 1}

    good = np.ones_like(scl, dtype=bool)
    for cls in bad_classes:
        good &= (scl != cls)
    return good


def scale_sr(dn: np.ndarray) -> np.ndarray:
    """
    Sentinel-2 L2A reflectance scale.
    """
    return dn.astype(np.float32) * cfg.SR_SCALE


def build_scene_products(
    item,
    glacier_id: str,
    geom_wgs84,
    geom_proj_cache: Optional[dict],
    roi_bounds,
    year: int,
):
    """
    Read one selected Sentinel-2 scene over one glacier polygon and return:
      ndsi, valid_mask, rgb_stack, profile, score

    Selection is metadata-based; this function only builds the cached outputs.
    """
    red_key = pick_band_asset(item, common_names=["red"])
    green_key = pick_band_asset(item, common_names=["green"])
    blue_key = pick_band_asset(item, common_names=["blue"])
    swir1_key = pick_band_asset(item, common_names=["swir16", "swir1"])

    if "SCL" not in item.assets:
        raise KeyError(f"SCL asset not found for item {item.id}")

    red_href = item.assets[red_key].href
    green_href = item.assets[green_key].href
    blue_href = item.assets[blue_key].href
    swir1_href = item.assets[swir1_key].href
    scl_href = item.assets["SCL"].href

    # Use SWIR1 native 20 m grid as the reference grid
    swir1_dn, profile = read_window_native(swir1_href, roi_bounds, out_dtype=np.float32)

    red_dn = read_window_match_grid(
        red_href,
        ref_profile=profile,
        out_dtype=np.float32,
        resampling=Resampling.bilinear,
    )

    green_dn = read_window_match_grid(
        green_href,
        ref_profile=profile,
        out_dtype=np.float32,
        resampling=Resampling.bilinear,
    )

    blue_dn = read_window_match_grid(
        blue_href,
        ref_profile=profile,
        out_dtype=np.float32,
        resampling=Resampling.bilinear,
    )

    scl = read_window_match_grid(
        scl_href,
        ref_profile=profile,
        out_dtype=np.uint8,
        resampling=Resampling.nearest,
    )

    crs_key = str(profile["crs"])
    if geom_proj_cache is not None and crs_key in geom_proj_cache:
        geom_proj = geom_proj_cache[crs_key]
    else:
        geom_proj = project_geometry_to_crs(geom_wgs84, profile["crs"])
        if geom_proj_cache is not None:
            geom_proj_cache[crs_key] = geom_proj

    inside = make_polygon_mask_from_projected_geom(geom_proj, profile)

    screen_good = sentinel_screen_mask(scl)
    compute_good = sentinel_compute_mask(scl)

    red = scale_sr(red_dn)
    green = scale_sr(green_dn)
    blue = scale_sr(blue_dn)
    swir1 = scale_sr(swir1_dn)

    denom = green + swir1
    ndsi = np.full_like(green, np.nan, dtype=np.float32)

    # Conservative valid-mask output
    screen_valid = inside & screen_good & np.isfinite(denom) & (np.abs(denom) > 1e-6)

    # Minimal mask for NDSI computation
    compute_valid = inside & compute_good & np.isfinite(denom) & (np.abs(denom) > 1e-6)
    ndsi[compute_valid] = (green[compute_valid] - swir1[compute_valid]) / denom[compute_valid]

    # RGB stack for later QC plotting
    rgb = np.stack([red, green, blue], axis=0).astype(np.float32)

    n_roi = int(np.sum(inside))
    n_valid = int(np.sum(screen_valid))
    valid_fraction = float(n_valid / n_roi) if n_roi > 0 else 0.0

    scene_dt = safe_item_datetime_str(item)
    cloud_cover = safe_cloud_cover(item)

    score = SceneScore(
        glacier_id=glacier_id,
        year=year,
        item_id=item.id,
        date=scene_dt,
        cloud_cover=cloud_cover,
        valid_fraction=valid_fraction,
        n_valid=n_valid,
        n_roi=n_roi,
    )

    return ndsi, screen_valid, rgb, profile, score


def write_scene_outputs(
    glacier_id: str,
    year: int,
    item_id: str,
    date_str: str,
    ndsi: np.ndarray,
    valid: np.ndarray,
    rgb: np.ndarray,
    profile: dict,
) -> tuple[str, str, str]:
    """
    Write NDSI, valid-mask, and RGB rasters for one accepted scene.
    """
    glacier_dir = cfg.CACHE_DIR / sanitize_id(glacier_id) / str(year)
    glacier_dir.mkdir(parents=True, exist_ok=True)

    safe_item = sanitize_id(item_id)
    safe_date = date_str[:10] if len(date_str) >= 10 else sanitize_id(date_str)

    ndsi_path = glacier_dir / f"ndsi_{safe_date}_{safe_item}.tif"
    valid_path = glacier_dir / f"validmask_{safe_date}_{safe_item}.tif"
    rgb_path = glacier_dir / f"rgb_{safe_date}_{safe_item}.tif"

    out_ndsi = np.where(np.isfinite(ndsi), ndsi, cfg.NODATA_FLOAT).astype(np.float32)
    out_valid = valid.astype(np.uint8)
    out_rgb = np.where(np.isfinite(rgb), rgb, cfg.NODATA_FLOAT).astype(np.float32)

    ndsi_profile = profile.copy()
    ndsi_profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        compress=cfg.GTIFF_COMPRESS,
        nodata=cfg.NODATA_FLOAT,
        tiled=True,
    )

    valid_profile = profile.copy()
    valid_profile.update(
        driver="GTiff",
        dtype="uint8",
        count=1,
        compress=cfg.GTIFF_COMPRESS,
        nodata=0,
        tiled=True,
    )

    rgb_profile = profile.copy()
    rgb_profile.update(
        driver="GTiff",
        dtype="float32",
        count=3,
        compress=cfg.GTIFF_COMPRESS,
        nodata=cfg.NODATA_FLOAT,
        tiled=True,
    )

    with rasterio.open(ndsi_path, "w", **ndsi_profile) as dst:
        dst.write(out_ndsi, 1)

    with rasterio.open(valid_path, "w", **valid_profile) as dst:
        dst.write(out_valid, 1)

    with rasterio.open(rgb_path, "w", **rgb_profile) as dst:
        dst.write(out_rgb)

    return str(ndsi_path), str(valid_path), str(rgb_path)


def process_one_glacier_one_year(
    catalog: Client,
    glacier_id: str,
    geom_wgs84,
    year: int,
    rows: list[dict],
) -> None:
    """
    Select the least-cloudy 3 scenes for one glacier-year using metadata only,
    then build the same cached products as before for those scenes.
    """
    roi_geom = geom_wgs84.__geo_interface__
    roi_bounds = tuple(gpd.GeoSeries([geom_wgs84], crs="EPSG:4326").total_bounds)

    items = stac_search_items(catalog, roi_geom, year)

    if not items:
        print(f"[{glacier_id}][{year}] No scenes found.")
        return

    items = maybe_debug_subset(items)
    selected_items = select_top_items_by_cloud(items, top_k=TOP_K_SCENES_PER_YEAR)

    if not selected_items:
        print(f"[{glacier_id}][{year}] No selectable scenes.")
        return

    print(
        f"[{glacier_id}][{year}] Selected {len(selected_items)} least-cloudy scene(s) "
        f"from {len(items)} candidate(s)."
    )

    signed_items = [pc.sign(it) for it in selected_items]
    geom_proj_cache: dict[str, object] = {}

    kept = 0

    for it in signed_items:
        try:
            ndsi, valid, rgb, profile, score = build_scene_products(
                item=it,
                glacier_id=glacier_id,
                geom_wgs84=geom_wgs84,
                geom_proj_cache=geom_proj_cache,
                roi_bounds=roi_bounds,
                year=year,
            )

            ndsi_path, valid_path, rgb_path = write_scene_outputs(
                glacier_id=glacier_id,
                year=year,
                item_id=score.item_id,
                date_str=score.date,
                ndsi=ndsi,
                valid=valid,
                rgb=rgb,
                profile=profile,
            )

            score.ndsi_path = ndsi_path
            score.validmask_path = valid_path
            score.rgb_path = rgb_path
            rows.append(asdict(score))

            kept += 1
            print(
                f"[{glacier_id}][{year}] keep   {score.item_id} | "
                f"date={score.date[:10]} | cloud={score.cloud_cover:.1f}% | "
                f"valid_fraction={score.valid_fraction:.3f} "
                f"({score.n_valid}/{score.n_roi})"
            )

        except Exception as e:
            print(f"[{glacier_id}][{year}] Skip item {it.id}: {e}")

    if kept == 0:
        print(f"[{glacier_id}][{year}] No accepted scenes written.")
    else:
        print(f"[{glacier_id}][{year}] Accepted {kept} scenes.")


def write_scene_index(rows: list[dict]) -> None:
    """
    Write scene index CSV for all accepted scenes.
    """
    cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = cfg.CACHE_DIR / "scene_index.csv"

    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values(["glacier_id", "year", "date", "item_id"]).reset_index(drop=True)
    else:
        df = pd.DataFrame(
            columns=[
                "glacier_id",
                "year",
                "item_id",
                "date",
                "cloud_cover",
                "valid_fraction",
                "n_valid",
                "n_roi",
                "ndsi_path",
                "validmask_path",
                "rgb_path",
            ]
        )

    df.to_csv(out_csv, index=False)
    print(f"Wrote scene index: {out_csv}")


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    rois = load_rois(cfg.ROI_PATH)
    catalog = Client.open(cfg.STAC_URL)

    cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for _, row in rois.iterrows():
        glacier_id = row["glacier_id"]
        geom = row.geometry

        for year in range(cfg.START_YEAR, cfg.END_YEAR + 1):
            process_one_glacier_one_year(
                catalog=catalog,
                glacier_id=glacier_id,
                geom_wgs84=geom,
                year=year,
                rows=rows,
            )

    write_scene_index(rows)


if __name__ == "__main__":
    main()