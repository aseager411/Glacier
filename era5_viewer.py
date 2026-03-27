import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

from datetime import datetime, timezone
from pathlib import Path
from shapely.geometry import box
from scipy.stats import theilslopes, kendalltau

gpd.options.io_engine = "fiona"


GLACIER_NAMES_N_TO_S = [
    "Jorge Montt Glacier",
    "O'higgins Glacier",
    "Grey Glacier",
]

GLACIER_COLORS = {
    "Jorge Montt Glacier": "tab:green",
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
    gdf = gpd.read_file(shapefile_path, engine="fiona")

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


def load_glacier_area_csv(csv_path):
    """
    Load glacier area measurements from CSV and convert to annual area series.

    Expected columns:
        GLACIER, AREA, DATE

    Returns:
        DataFrame with columns:
            year, glacier_name, area_km2
    """
    df = pd.read_csv(csv_path)

    required_cols = {"GLACIER", "AREA", "DATE"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Area CSV missing required columns: {missing}")

    df = df.copy()
    df = df.dropna(subset=["GLACIER", "AREA", "DATE"])

    glacier_map = {
        "GREY": "Grey Glacier",
        "O´HIGGINS": "O'higgins Glacier",
        "O'HIGGINS": "O'higgins Glacier",
        "JORGE MONTT": "Jorge Montt Glacier",
    }

    df["GLACIER"] = df["GLACIER"].astype(str).str.strip().str.upper()
    df["glacier_name"] = df["GLACIER"].map(glacier_map)

    if df["glacier_name"].isna().any():
        unknown = sorted(df.loc[df["glacier_name"].isna(), "GLACIER"].unique())
        print(f"Warning: unrecognized glacier names in area CSV: {unknown}")

    df = df.dropna(subset=["glacier_name"])

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"])

    df["year"] = df["DATE"].dt.year.astype(int)
    df["area_km2"] = pd.to_numeric(df["AREA"], errors="coerce")
    df = df.dropna(subset=["area_km2"])

    area_df = (
        df.groupby(["glacier_name", "year"], as_index=False)
        .agg(area_km2=("area_km2", "mean"))
        .sort_values(["glacier_name", "year"])
    )

    return area_df


def load_gee_glacier_area_csv(csv_path):
    """
    Load glacier area output exported from Google Earth Engine.

    Expected relevant columns:
        box_name, year, glacier_area_m2

    Notes:
    - Keeps only year and area
    - Collapses the 3 duplicate scene rows per glacier-year to one value
    - Converts m^2 to km^2
    """
    df = pd.read_csv(csv_path)

    required_cols = {"box_name", "year", "glacier_area_m2"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"GEE CSV missing required columns: {missing}")

    df = df.copy()
    df = df.dropna(subset=["box_name", "year", "glacier_area_m2"])

    glacier_map = {
        "DE_MONAR": "Jorge Montt Glacier",
        "GREY": "Grey Glacier",
        "O'HIGGINS": "O'higgins Glacier",
    }

    df["box_name"] = df["box_name"].astype(str).str.strip().str.upper()
    df["glacier_name"] = df["box_name"].map(glacier_map)

    if df["glacier_name"].isna().any():
        unknown = sorted(df.loc[df["glacier_name"].isna(), "box_name"].unique())
        print(f"Warning: unrecognized glacier names in GEE CSV: {unknown}")

    df = df.dropna(subset=["glacier_name"])

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["glacier_area_m2"] = pd.to_numeric(df["glacier_area_m2"], errors="coerce")
    df = df.dropna(subset=["year", "glacier_area_m2"])

    gee_area_df = (
        df.groupby(["glacier_name", "year"], as_index=False)
        .agg(glacier_area_m2=("glacier_area_m2", "mean"))
        .sort_values(["glacier_name", "year"])
    )

    gee_area_df["year"] = gee_area_df["year"].astype(int)
    gee_area_df["area_km2"] = gee_area_df["glacier_area_m2"] / 1e6

    return gee_area_df[["glacier_name", "year", "area_km2"]]


def scale_gee_area_to_legacy(legacy_area_df, gee_area_df):
    """
    Scale each glacier's GEE area series so that its first value (2019)
    matches the last value from the legacy area series (assumed to be 2016).

    Returns:
        scaled GEE dataframe with columns:
            glacier_name, year, area_km2
    """
    legacy_area_df = legacy_area_df.copy().sort_values(["glacier_name", "year"])
    gee_area_df = gee_area_df.copy().sort_values(["glacier_name", "year"])

    scaled_parts = []

    for glacier_name in gee_area_df["glacier_name"].unique():
        legacy_sub = legacy_area_df[legacy_area_df["glacier_name"] == glacier_name].sort_values("year")
        gee_sub = gee_area_df[gee_area_df["glacier_name"] == glacier_name].sort_values("year").copy()

        if legacy_sub.empty:
            print(f"Warning: no legacy area data found for {glacier_name}; leaving GEE values unscaled.")
            scaled_parts.append(gee_sub)
            continue

        if gee_sub.empty:
            continue

        legacy_anchor = legacy_sub.iloc[-1]["area_km2"]
        gee_anchor = gee_sub.iloc[0]["area_km2"]

        if not np.isfinite(legacy_anchor) or not np.isfinite(gee_anchor) or gee_anchor == 0:
            print(f"Warning: could not scale GEE series for {glacier_name}; leaving values unscaled.")
            scaled_parts.append(gee_sub)
            continue

        scale_factor = legacy_anchor / gee_anchor
        gee_sub["area_km2"] = gee_sub["area_km2"] * scale_factor
        scaled_parts.append(gee_sub)

    if not scaled_parts:
        return pd.DataFrame(columns=["glacier_name", "year", "area_km2"])

    scaled_df = pd.concat(scaled_parts, ignore_index=True)
    return scaled_df.sort_values(["glacier_name", "year"]).reset_index(drop=True)


def combine_area_sources(legacy_area_df, gee_area_df_scaled):
    """
    Combine the original area series with the scaled GEE area series.
    """
    combined = pd.concat(
        [
            legacy_area_df[["glacier_name", "year", "area_km2"]],
            gee_area_df_scaled[["glacier_name", "year", "area_km2"]],
        ],
        ignore_index=True,
    )

    combined = (
        combined.dropna(subset=["glacier_name", "year", "area_km2"])
        .drop_duplicates(subset=["glacier_name", "year"], keep="last")
        .sort_values(["glacier_name", "year"])
        .reset_index(drop=True)
    )

    return combined


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

def compute_sen_slope_and_p(years, values):
    """
    Compute Sen's slope, intercept, and a two-sided p-value for monotonic trend.

    Uses:
    - Sen's slope via scipy.stats.theilslopes
    - p-value via Kendall tau test

    Returns:
        slope, intercept, p_value
    """
    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan, np.nan, np.nan

    if np.all(y == y[0]):
        return 0.0, y[0], 1.0

    slope, intercept, _, _ = theilslopes(y, x, alpha=0.95)
    p_value = kendalltau(x, y).pvalue

    return slope, intercept, p_value

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

    if area_df is None:
        area_df = get_placeholder_area_series(years, glacier_order)

    area_df = area_df.copy()
    area_df = area_df.sort_values(["glacier_name", "year"])

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

            if var != "area_km2":
                ax.plot(
                    plot_df["year"],
                    plot_df[var],
                    marker="o",
                    color=line_color,
                    linewidth=2,
                    markersize=4,
                )
            else:
                legacy_area_sub = area_sub[area_sub["year"] <= 2016].sort_values("year")
                gee_area_sub = area_sub[area_sub["year"] >= 2019].sort_values("year")

                if not legacy_area_sub.empty:
                    ax.plot(
                        legacy_area_sub["year"],
                        legacy_area_sub["area_km2"],
                        marker="o",
                        color=line_color,
                        linewidth=2,
                        markersize=4,
                    )

                if not gee_area_sub.empty:
                    ax.plot(
                        gee_area_sub["year"],
                        gee_area_sub["area_km2"],
                        marker="s",   # square marker for new GEE-based series
                        color=line_color,
                        linewidth=2,
                        markersize=5,
                    )


            if var != "area_km2":
                trend_years = plot_df["year"].values
                trend_vals = plot_df[var].values
            else:
                trend_years = area_sub["year"].values
                trend_vals = area_sub["area_km2"].values

            slope, intercept, p_value = compute_sen_slope_and_p(trend_years, trend_vals)

            if np.isfinite(slope) and np.isfinite(intercept):
                x_trend = np.asarray(trend_years, dtype=float)
                y_trend = slope * x_trend + intercept

                ax.plot(
                    x_trend,
                    y_trend,
                    linestyle="--",
                    color="black",
                    linewidth=1.5,
                    alpha=0.8,
                )

            if np.isfinite(slope) and np.isfinite(p_value):
                trend_text = f"Sen slope = {slope:.3f}\np = {p_value:.3f}"
            else:
                trend_text = "Sen slope = NA\np = NA"

            ax.text(
                0.03, 0.97,
                trend_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
            )



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

    era_file = repo_root / "data" / "ERA" / "ERA5_new.grib"
    shapefile = repo_root / "data" / "roi" / "spi_glaciers.shp"
    area_csv = repo_root / "data" / "Glacier_Areas" / "SPI_Glacier_Areas.csv"
    gee_area_csv = repo_root / "data" / "Glacier_Areas" / "spi_glacier_run0.csv"

    output_dir = repo_root / "outputs" / "era5_glacier_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    times, temp, precip, lats, lons, transform, crs = load_era5_grib(era_file)
    glaciers_gdf = load_and_name_glaciers(shapefile, crs)

    pixel_polygons = build_pixel_polygons(lats, lons, transform)
    glacier_weights = compute_glacier_weights(glaciers_gdf, pixel_polygons)

    monthly_df = compute_glacier_timeseries(times, temp, precip, glacier_weights)
    annual_df = aggregate_to_annual(monthly_df)

    legacy_area_df = load_glacier_area_csv(area_csv)
    gee_area_df = load_gee_glacier_area_csv(gee_area_csv)
    gee_area_df_scaled = scale_gee_area_to_legacy(legacy_area_df, gee_area_df)
    area_df = combine_area_sources(legacy_area_df, gee_area_df_scaled)

    plot_annual_grid(
        annual_df,
        output_dir / "annual_glacier_grid_3x3.png",
        area_df=area_df,
    )