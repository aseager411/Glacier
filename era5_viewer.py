import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

from datetime import datetime, timezone
from pathlib import Path
from shapely.geometry import box


GLACIER_NAMES_N_TO_S = [
    "De Monar Glacier",
    "O'higgins Glacier",
    "Grey Glacier",
]

GLACIER_COLORS = {
    "De Monar": "tab:green",
    "O'higgins Glacier": "tab:orange",
    "Grey Glacier": "tab:purple",
}


def load_era5_grib(path):
    """
    Load monthly ERA5 GRIB with 2T and TP bands.

    Returns:
        times, temp[C], precip[mm], lats, lons, transform, crs
    """
    with rasterio.open(path) as src:
        temp_list = []
        precip_list = []
        times = []

        for band in src.indexes:
            tags = src.tags(band)
            element = tags["GRIB_ELEMENT"]
            valid_time = int(tags["GRIB_VALID_TIME"])
            arr = src.read(band)

            if element == "2T":
                temp_list.append(arr)
                times.append(datetime.fromtimestamp(valid_time, tz=timezone.utc))
            elif element == "TP":
                precip_list.append(arr)

        temp = np.stack(temp_list) - 273.15   # K -> C
        precip = np.stack(precip_list) * 1000.0   # m -> mm

        transform = src.transform
        width = src.width
        height = src.height
        crs = src.crs

        lons = np.array([transform.c + transform.a * (c + 0.5) for c in range(width)])
        lats = np.array([transform.f + transform.e * (r + 0.5) for r in range(height)])

    return times, temp, precip, lats, lons, transform, crs


def load_and_name_glaciers(shapefile_path, target_crs):
    """
    Load shapefile, reproject if needed, and assign glacier names
    by centroid latitude from north to south.
    """
    gdf = gpd.read_file(shapefile_path)

    if gdf.empty:
        raise ValueError(f"No features found in shapefile: {shapefile_path}")

    if gdf.crs is None:
        raise ValueError("Shapefile has no CRS defined.")

    if target_crs is not None and gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    gdf = gdf.copy().reset_index(drop=True)
    gdf["centroid_y"] = gdf.geometry.centroid.y
    gdf = gdf.sort_values("centroid_y", ascending=False).reset_index(drop=True)

    if len(gdf) != 3:
        print(
            f"Warning: expected 3 glacier polygons, found {len(gdf)}. "
            "Assigning names to the first three in north-to-south order."
        )

    gdf["glacier_name"] = GLACIER_NAMES_N_TO_S[: len(gdf)]
    return gdf.drop(columns=["centroid_y"])


def build_pixel_polygons(lats, lons, transform):
    """
    Build shapely polygons for each ERA5 grid cell once.
    """
    dx = abs(transform.a)
    dy = abs(transform.e)
    half_dx = dx / 2.0
    half_dy = dy / 2.0

    nrows = len(lats)
    ncols = len(lons)
    pixel_polygons = np.empty((nrows, ncols), dtype=object)

    for r, lat in enumerate(lats):
        for c, lon in enumerate(lons):
            pixel_polygons[r, c] = box(
                lon - half_dx,
                lat - half_dy,
                lon + half_dx,
                lat + half_dy,
            )

    return pixel_polygons


def compute_glacier_weights(gdf, pixel_polygons):
    """
    Compute one area-overlap weight grid per glacier polygon.
    Weights are normalized to sum to 1 over overlapping ERA5 cells.
    """
    nrows, ncols = pixel_polygons.shape
    weights = {}

    for _, row in gdf.iterrows():
        glacier_name = row["glacier_name"]
        geom = row.geometry
        geom_bounds = geom.bounds

        w = np.zeros((nrows, ncols), dtype=float)

        for r in range(nrows):
            for c in range(ncols):
                cell = pixel_polygons[r, c]
                cb = cell.bounds

                bbox_overlap = not (
                    cb[2] < geom_bounds[0]
                    or cb[0] > geom_bounds[2]
                    or cb[3] < geom_bounds[1]
                    or cb[1] > geom_bounds[3]
                )

                if bbox_overlap and cell.intersects(geom):
                    overlap_area = cell.intersection(geom).area
                    if overlap_area > 0:
                        w[r, c] = overlap_area / cell.area

        if w.sum() == 0:
            print(f"Warning: {glacier_name} does not overlap any ERA5 cells.")
        else:
            w = w / w.sum()

        weights[glacier_name] = w

    return weights


def compute_glacier_timeseries(times, temp, precip, weights):
    """
    Monthly glacier-weighted temp/precip from overlapping ERA5 cells.
    Excludes year 2026.
    """
    records = []

    for glacier_name, w in weights.items():
        if np.all(w == 0):
            continue

        temp_series = (temp * w[None, :, :]).sum(axis=(1, 2))
        precip_series = (precip * w[None, :, :]).sum(axis=(1, 2))

        for t, tval, pval in zip(times, temp_series, precip_series):
            if t.year == 2026:
                continue

            records.append(
                {
                    "time": t,
                    "year": t.year,
                    "glacier_name": glacier_name,
                    "temp_C": tval,
                    "precip_mm": pval,
                }
            )

    return pd.DataFrame(records)


def get_placeholder_area_series(years, glacier_names):
    """
    Placeholder annual glacier area table.
    Replace later with real area data.
    """
    records = []
    for glacier_name in glacier_names:
        for year in years:
            records.append(
                {
                    "year": year,
                    "glacier_name": glacier_name,
                    "area_km2": np.nan,
                }
            )
    return pd.DataFrame(records)


def aggregate_to_annual(monthly_df):
    """
    Annualize the monthly climate series:
    - temperature = annual mean of monthly means
    - precipitation = annual sum of monthly totals
    """
    annual = (
        monthly_df.groupby(["glacier_name", "year"], as_index=False)
        .agg(
            temp_C=("temp_C", "mean"),
            precip_mm=("precip_mm", "sum"),
        )
        .sort_values(["glacier_name", "year"])
    )
    return annual


def plot_annual_grid(annual_df, save_path, area_df=None):
    """
    Make one 3x3 figure:
    columns = glaciers
    rows = variables (temp, precip, area)
    Each panel = one glacier + one variable over time.

    Only left column gets y-axis labels.
    Only bottom row gets x-axis labels.
    """
    glacier_order = [g for g in GLACIER_NAMES_N_TO_S if g in annual_df["glacier_name"].unique()]
    years = np.sort(annual_df["year"].unique())

    # compute common y-limits for each variable row
    def safe_ylim(series, pad_frac=0.05):
        vals = np.asarray(series, dtype=float)
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            return None

        vmin = vals.min()
        vmax = vals.max()

        if vmin == vmax:
            pad = 1.0 if vmin == 0 else abs(vmin) * pad_frac
        else:
            pad = (vmax - vmin) * pad_frac

        return (vmin - pad, vmax + pad)


    temp_ylim = safe_ylim(annual_df["temp_C"])
    precip_ylim = safe_ylim(annual_df["precip_mm"])
    area_ylim = safe_ylim(area_df["area_km2"])

    if area_df is None:
        area_df = get_placeholder_area_series(years, glacier_order)

    area_df = area_df.copy()
    area_df = area_df.sort_values(["glacier_name", "year"])

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex="col")

    row_info = [
        ("temp_C", "Annual mean temp (°C)", "tab:red"),
        ("precip_mm", "Annual precip (mm)", "tab:blue"),
        ("area_km2", "Area (km²)", "tab:gray"),
    ]

    for col_idx, glacier_name in enumerate(glacier_order):
        glacier_sub = annual_df[annual_df["glacier_name"] == glacier_name].sort_values("year")
        area_sub = area_df[area_df["glacier_name"] == glacier_name].sort_values("year")

        for row_idx, (var, ylabel, line_color) in enumerate(row_info):
            ax = axes[row_idx, col_idx]

            if var in glacier_sub.columns:
                plot_df = glacier_sub
            else:
                plot_df = area_sub

            ax.plot(
                plot_df["year"],
                plot_df[var],
                marker="o",
                color=line_color,
                linewidth=2,
                markersize=4,
            )

            # enforce identical y-axis limits across glaciers
            if var == "temp_C" and temp_ylim is not None:
                ax.set_ylim(temp_ylim)
            elif var == "precip_mm" and precip_ylim is not None:
                ax.set_ylim(precip_ylim)
            elif var == "area_km2" and area_ylim is not None:
                ax.set_ylim(area_ylim)

            ax.grid(True, alpha=0.3)

            if row_idx == 0:
                ax.set_title(glacier_name)

            if col_idx == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel("")
                ax.tick_params(left=False, labelleft=False)

            if row_idx == 2:
                ax.set_xlabel("Year")
            else:
                ax.tick_params(labelbottom=False)

    fig.suptitle("Annual glacier climate and area time series", y=0.98)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent

    era_file = repo_root / "data" / "ERA" / "ERA5_SPI_monthly.grib"
    shapefile = repo_root / "data" / "roi" / "spi_glaciers.shp"

    output_dir = repo_root / "outputs" / "era5_glacier_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    times, temp, precip, lats, lons, transform, crs = load_era5_grib(era_file)
    glaciers_gdf = load_and_name_glaciers(shapefile, crs)

    pixel_polygons = build_pixel_polygons(lats, lons, transform)
    glacier_weights = compute_glacier_weights(glaciers_gdf, pixel_polygons)
    
    monthly_df = compute_glacier_timeseries(times, temp, precip, glacier_weights)
    annual_df = aggregate_to_annual(monthly_df)

    # Placeholder annual area
    area_df = get_placeholder_area_series(
        years=np.sort(annual_df["year"].unique()),
        glacier_names=[g for g in GLACIER_NAMES_N_TO_S if g in annual_df["glacier_name"].unique()],
    )

    plot_annual_grid(
        annual_df,
        output_dir / "annual_glacier_grid_3x3.png",
        area_df=area_df,
    )