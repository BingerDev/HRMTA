"""
Main configuration file for HRMTA.
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Load the .env file
load_dotenv(PROJECT_ROOT / ".env")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR = PROJECT_ROOT / "inputs" / "input-PL"
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = PROJECT_ROOT / "cache"

# Model run output folder
KEEP_RUN_HISTORY = True

if KEEP_RUN_HISTORY:
    RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    RUN_OUTPUT_DIR = OUTPUT_DIR / RUN_TIMESTAMP
else:
    RUN_OUTPUT_DIR = OUTPUT_DIR

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
CACHE_DIR.mkdir(exist_ok=True, parents=True)
RUN_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Engine mode
MODE = "pro"

# Data tier
DATA_TIER = "lite"

# Core - always loaded (minimum feature set)
_RASTERS_CORE = {
    "dem": INPUT_DIR / "copernicus_dem.tif",
}

# Terrain physics
_RASTERS_TERRAIN = {
    "tpi_500": INPUT_DIR / "tpi_500.tif",
    "tpi_2000": INPUT_DIR / "tpi_2000.tif",
    "svf": INPUT_DIR / "svf.tif",
    "cap_anomaly": INPUT_DIR / "cap_anomaly.tif",
    "hand": INPUT_DIR / "hand.tif",
}

# Environment
_RASTERS_ENVIRONMENT = {
    "landscan": INPUT_DIR / "landscan_100m.tif",
    "land_cover": INPUT_DIR / "worldcover_100m.tif",
    "settlement": INPUT_DIR / "built_up_fraction.tif",
    "imperviousness": INPUT_DIR / "imperviousness_fraction.tif",
    "forests": INPUT_DIR / "tree_fraction.tif",
    "lst_summer": INPUT_DIR / "lst_summer.tif",
    "water": INPUT_DIR / "water_fraction.tif",
    "canopy_height": INPUT_DIR / "canopy_height.tif",
    "building_height": INPUT_DIR / "building_height.tif",
    "cropland": INPUT_DIR / "cropland_fraction.tif",
}

# Build RASTER_FILES based on selected tier
_TIER_GROUPS = {
    "lite":     [_RASTERS_CORE],
    "standard": [_RASTERS_CORE, _RASTERS_TERRAIN],
    "full":     [_RASTERS_CORE, _RASTERS_TERRAIN, _RASTERS_ENVIRONMENT],
}

if DATA_TIER not in _TIER_GROUPS:
    print(f"⚠️  Unknown DATA_TIER '{DATA_TIER}', falling back to 'full'")
    DATA_TIER = "full"

RASTER_FILES = {}
for group in _TIER_GROUPS[DATA_TIER]:
    RASTER_FILES.update(group)

SHAPEFILE = INPUT_DIR / "poland.shp"
COLOR_SCALE = INPUT_DIR / "color_scale.csv"
GEOCACHE_FILE = CACHE_DIR / "geocoding_cache.json"

# Data sources & filtering

# IMGW
IMGW_PROVINCES = list(range(1, 17))
IMGW_DATA_MODE = "all"

TRAX_REGION_IDS = [35, 4, 48, 12, 8, 3, 37, 6, 10, 88, 11, 7, 74, 5, 9, 39]

NETATMO_CONFIG = {
    "lat_ne": 55.4, "lon_ne": 24.8,
    "lat_sw": 48.4, "lon_sw": 13.7,
    "access_token": os.getenv("NETATMO_TOKEN", "YOUR_TOKEN_HERE")
}

EDWIN_CONFIG = {
    "api_base": "https://edwin-meteo.apps.paas.psnc.pl",
    "station_types": ["WEATHER"],
    "workers": 20,
    "lookback_hours": 2,
}

# Station deduplication radius
PWS_DEDUP_RADIUS_M = 100

# NWP (Numerical Weather Prediction) settings

# NWP config
NWP_CONFIG = {
    "source": "HARMONIE-DMI",
    "stac_endpoint": "https://opendataapi.dmi.dk/v1/forecastdata/collections/harmonie_dini_sf/items",
    "cache_hours": 3,
}

# ICON-EU settings
ICON_CONFIG = {
    "base_url": "https://opendata.dwd.de/weather/nwp/icon-eu/grib",
    "variables": ["t_2m", "clct", "u_10m", "v_10m", "hsurf", "t_850"],
    "cache_hours": 3,
}

# NWP cache management (maximum GRIB/NetCDF files to keep per NWP source)
NWP_CACHE_MAX_FILES = 2

# Spatial Quality Control
PERFORM_SPATIAL_QC = True

# Source priors
QC_SOURCE_PRIORS = {
    'IMGW': 1.0, 'EDWIN': 0.95,
    'TRAX': 0.7,
    'NETATMO': 0.5,
}

# Source-specific tolerance midpoints
QC_SOURCE_TOLERANCES = {
    'IMGW': 3.5, 'EDWIN': 3.0,
    'TRAX': 2.5,
    'NETATMO': 2.0,
}

# FS-ISCT hyperparameters
QC_HARD_REJECT_THRESHOLD = 8.0
QC_CONFIDENCE_MIN = 0.01
QC_BUTTERWORTH_ORDER = 4
QC_ITERATIONS = 3
QC_ANCHOR_WEIGHT = 0.5

# Spatial declustering
QC_DECLUSTER_RADIUS_KM = 2.0

# Feature-space kernel length scales
QC_KERNEL_GEO_KM = 30.0
QC_KERNEL_DEM_M = 200.0
QC_KERNEL_CAP = 1.5
QC_KERNEL_SETTLEMENT = 0.2
QC_KERNEL_SVF = 0.15

# Isolation tolerance expansion
QC_ISOLATION_ALPHA = 1.0

# Neighbor search
QC_MAX_NEIGHBORS = 80
QC_MAX_SEARCH_RADIUS_KM = 50.0

# Backward compat aliases
QC_NEIGHBORS = 10
QC_Z_THRESHOLD = 3.0
QC_ABS_THRESHOLD = 3.5
QC_SOURCE_THRESHOLDS = QC_SOURCE_TOLERANCES

# Lapse rate settings
STANDARD_LAPSE_RATE = 0.0065
USE_DYNAMIC_LAPSE_RATE = True
MIN_STATIONS_FOR_DYNAMIC_LR = 15
MIN_ELEVATION_SPREAD = 200

# Coordinate systems definition
CRS_WGS84 = "EPSG:4326"
CRS_POLAND = "EPSG:2180"

# Interpolation extent settings
INTERPOLATION_REGION = "Poland" # Region name PL/EN (like "Mazowieckie" / "Masovian")
REGIONAL_BUFFER_KM = 40
MIN_REGIONAL_STATIONS = 30
VOIVODESHIP_SHAPEFILE = INPUT_DIR / "poland_voivodeships.shp"
COUNTIES_SHAPEFILE = INPUT_DIR / "poland_counties.shp"
DISPLAY_COUNTIES = True

# Model settings
GRID_RESOLUTION = 1000  # meters

# Tier-aware resolution guidance
_TIER_MIN_RESOLUTION = {"lite": 1000, "standard": 250, "full": 100}
_tier_min = _TIER_MIN_RESOLUTION.get(DATA_TIER, 100)
if GRID_RESOLUTION < _tier_min:
    print(f"⚠️  GRID_RESOLUTION={GRID_RESOLUTION}m with DATA_TIER='{DATA_TIER}' - "
          f"recommended minimum is {_tier_min}m (insufficient raster detail below this)")
del _tier_min

# Spatial CV
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42
SPATIAL_BUFFER_KM = 10

# Architecture: Robust spatial-physics stacking

# Stage 1: HuberRegressor
TREND_FEATURES = ['dem', 'y_pl', 'x_pl', 'y_pl_sq', 'x_pl_sq', 'x_y_interaction'] # Elevation + Linear coords + Quadratic surface

# Ensemble settings
USE_ENSEMBLE = True
ENSEMBLE_N_MODELS = 5
ENSEMBLE_SEEDS = [42, 123, 456, 789, 101112]

# Stage 2: LightGBM
LIGHTGBM_PARAMS = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 2500,
    'learning_rate': 0.008,
    'num_leaves': 24,
    'max_depth': 6,
    'min_child_samples': 30,
    'subsample': 0.65,
    'colsample_bytree': 0.6,
    'reg_alpha': 1.5,
    'reg_lambda': 3.0,
    'feature_fraction_bynode': 0.5,
    'n_jobs': -1,
    'verbose': -1,
    'random_state': RANDOM_STATE,
}

# Stage 3: Residual Kriging
USE_RESIDUAL_KRIGING = True
RESIDUAL_KRIGING_VARIOGRAM = "exponential"

# Feature engineering settings
EXTRACT_TERRAIN_DERIVATIVES = True
TERRAIN_WINDOW_SIZES = [9, 27]
CREATE_FEATURE_INTERACTIONS = True
COMPUTE_DISTANCE_FEATURES = True
DISTANCE_FEATURES = {'coast': True, 'mountains': True}
USE_SPATIAL_LAG_FEATURES = False
SPATIAL_LAG_NEIGHBORS = 5

# Interaction pairs
INTERACTION_PAIRS = [
    ('dem', 'aspect_sin'),
    ('dem', 'aspect_cos'),
    ('dem', 'slope'),
    ('lst_summer', 'settlement'),
    ('dist_coast', 'dem'),
]

# Post-processing
APPLY_SMOOTHING = True
SMOOTHING_SIGMA = 1.0

# Visualization overlays
DISPLAY_CONTOURS = True
CONTOUR_INTERVAL = 1.0

# Clean mode, removes title, source footer, and Tmax/Tmin callouts
# NOTE: Enabling this removes the observational data source attribution footer.
# If you publish clean mode output publicly, ensure you provide attribution
# for the observational data sources separately.
DISPLAY_CLEAN_MODE = False

# Visualization
DISPLAY_STATION_SOURCES = ["IMGW"]
DISPLAY_OBSERVATIONS_ONLY = True
OUTPUT_PLOT = RUN_OUTPUT_DIR / "temperature_map.png"
OUTPUT_UNCERTAINTY = RUN_OUTPUT_DIR / "uncertainty_map.png"
DPI = 300

# Environment
proj = Path(sys.prefix) / 'share' / 'proj'
if proj.exists():
    os.environ['PROJ_LIB'] = str(proj)
