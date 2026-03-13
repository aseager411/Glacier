"""
config.py

Configuration for glacier Sentinel-2 preprocessing pipeline.
All global parameters for build_cache.py live here so that
changing thresholds or time windows does not require editing code.
"""

from pathlib import Path


# ---------------------------------------------------
# Directory structure
# ---------------------------------------------------

ROOT = Path(".")
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"

# Glacier ROI polygons (GeoJSON or shapefile)
ROI_PATH = DATA_DIR / "roi" / "spi_glaciers.shp"


# ---------------------------------------------------
# Planetary Computer / STAC configuration
# ---------------------------------------------------

# Microsoft Planetary Computer STAC endpoint
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Sentinel-2 Level-2A surface reflectance
COLLECTION = "sentinel-2-l2a"


# ---------------------------------------------------
# Time range
# ---------------------------------------------------

# Years to process
START_YEAR = 2024
END_YEAR = 2024

# Seasonal window (late summer / minimum seasonal snow)
# Adjust depending on hemisphere
START_MM_DD = "02-01"
END_MM_DD = "03-01"


# ---------------------------------------------------
# Scene filtering
# ---------------------------------------------------

# Scene-level cloud cover threshold
# This is only a coarse prefilter — real filtering
# happens using SCL pixel classes.
MAX_CLOUD = 80

# Minimum usable glacier pixels required
MIN_ROI_PIXELS = 50

# Minimum fraction of glacier pixels that must be valid
# (clear, not cloud, not shadow)
MIN_VALID_FRACTION = 0.20


# ---------------------------------------------------
# Pixel masking options
# ---------------------------------------------------

# Mask water pixels (SCL class 6)
MASK_WATER = False

# Mask snow/ice pixels (SCL class 11)
# Usually FALSE for glacier studies
MASK_SNOW_ICE = False


# ---------------------------------------------------
# Sentinel-2 reflectance scaling
# ---------------------------------------------------

# Sentinel-2 L2A scale factor
# DN * 1e-4 → reflectance
SR_SCALE = 1e-4


# ---------------------------------------------------
# Output GeoTIFF configuration
# ---------------------------------------------------

# Compression method
GTIFF_COMPRESS = "deflate"

# Floating-point nodata value
NODATA_FLOAT = -9999.0


# ---------------------------------------------------
# Debug / development options
# ---------------------------------------------------

# Enable debugging mode
DEBUG = False

# Maximum number of scenes processed per glacier/year in debug
DEBUG_MAX_SCENES = 8

# Sort scenes by cloud cover before selecting debug subset
DEBUG_SORT_BY_CLOUD = True


# ---------------------------------------------------
# Optional behavior
# ---------------------------------------------------

# Write metadata CSV listing all accepted scenes
WRITE_SCENE_INDEX = True


# ---------------------------------------------------
# Derived paths (do not modify)
# ---------------------------------------------------

SCENE_INDEX_PATH = CACHE_DIR / "scene_index.csv"

# ---------------------------------------------------
# Analysis outputs
# ---------------------------------------------------

RESULTS_DIR = DATA_DIR / "results"
RESULTS_PER_GLACIER_DIR = RESULTS_DIR / "per_glacier"
RESULTS_SUMMARY_DIR = RESULTS_DIR / "summary"

# Threshold used by analyze.py to convert NDSI to glacier mask
NDSI_THRESHOLD = 0.60

# Optional outputs
WRITE_ANNUAL_SUMMARY = True
WRITE_QC_SCENE_TABLE = True
# ------
## QC plotting
# ------
# ---------------------------------------------------
# QC plot directories
# ---------------------------------------------------
MAKE_SCENE_QC_PANELS = True
MAKE_QC_TIMESERIES = True
MAKE_QC_STRIPS = True

RGB_STRETCH_LOW = 2
RGB_STRETCH_HIGH = 98
GLACIER_OVERLAY_ALPHA = 0.35

# Set to an integer like 10 for a quick test, or None for all scenes
QC_MAX_SCENES = 10

QC_STRIP_MAX_SCENES = None
QC_DPI = 150


QC_PLOTS_DIR = RESULTS_DIR / "qc"

QC_SCENES_DIR = QC_PLOTS_DIR / "scenes"
QC_TIMESERIES_DIR = QC_PLOTS_DIR / "timeseries"
QC_STRIPS_DIR = QC_PLOTS_DIR / "strips"

