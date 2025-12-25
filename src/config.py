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
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_OUTPUT_DIR = OUTPUT_DIR / RUN_TIMESTAMP

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
CACHE_DIR.mkdir(exist_ok=True, parents=True)
RUN_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Data files
RASTER_FILES = {
    "dem": INPUT_DIR / "copernicus_dem.tif",
    "landscan": INPUT_DIR / "landscan_hd.tif",
    "land_cover": INPUT_DIR / "land_cover.tif",
    "settlement": INPUT_DIR / "human_settlement.tif",
    "forests": INPUT_DIR / "forests.tif",
    "ecostress": INPUT_DIR / "raw_ecostress.tif",
    "water": INPUT_DIR / "water_bodies.tif",
}

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

# Spatial Outlier detection settings
PERFORM_SPATIAL_QC = True
QC_NEIGHBORS = 10
QC_Z_THRESHOLD = 3.0
QC_ABS_THRESHOLD = 3.5
QC_LAPSE_RATE = 0.0065

# Coordinate systems definition
CRS_WGS84 = "EPSG:4326"
CRS_POLAND = "EPSG:2180"

# Model evaluation settings
GRID_RESOLUTION = 5000  # meters

# Spatial CV
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42
SPATIAL_BUFFER_KM = 20

# Architecture: Robust spatial-physics stacking

# Stage 1: HuberRegressor
TREND_FEATURES = ['dem', 'y_pl', 'x_pl'] # Elevation + Latitude + Longitude

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
    'num_leaves': 16,
    'max_depth': 5,
    'min_child_samples': 60,
    'subsample': 0.65,
    'colsample_bytree': 0.6,
    'reg_alpha': 3.5,
    'reg_lambda': 6.0,
    'feature_fraction_bynode': 0.7,
    'n_jobs': -1,
    'verbose': -1,
    'random_state': RANDOM_STATE,
}

# Stage 3: Residual Kriging
USE_RESIDUAL_KRIGING = True
RESIDUAL_KRIGING_VARIOGRAM = "exponential"

# Feature engineering settings
EXTRACT_TERRAIN_DERIVATIVES = True
TERRAIN_WINDOW_SIZES = [3, 9, 27]
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
    ('ecostress', 'settlement'),
    ('dist_coast', 'dem'),
]

# Post-processing
APPLY_SMOOTHING = True
SMOOTHING_SIGMA = 1.0

# Visualization
DISPLAY_STATION_SOURCES = ["IMGW"]
OUTPUT_PLOT = RUN_OUTPUT_DIR / "temperature_map.png"
OUTPUT_UNCERTAINTY = RUN_OUTPUT_DIR / "uncertainty_map.png"
DPI = 300

# Environment
proj = Path(sys.prefix) / 'share' / 'proj'
if proj.exists():
    os.environ['PROJ_LIB'] = str(proj)
