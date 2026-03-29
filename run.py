"""
HRMTA, main production pipeline.
Integrates Spatial QC, Robust Physics-Stacking, and rigorous evaluation.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from pyproj import Transformer
from sklearn.model_selection import train_test_split
from scipy.spatial import cKDTree
import warnings
import json
import gc
from datetime import datetime, timezone
from pathlib import Path
import glob as glob_module
import rasterio
from rasterio.transform import rowcol

from src.config import (
    GRID_RESOLUTION, CRS_POLAND, CRS_WGS84,
    TEST_SIZE, VAL_SIZE, RANDOM_STATE,
    OUTPUT_DIR, RUN_OUTPUT_DIR, SPATIAL_BUFFER_KM,
    USE_ENSEMBLE, ENSEMBLE_N_MODELS, ENSEMBLE_SEEDS,
    TREND_FEATURES, LIGHTGBM_PARAMS,
    OUTPUT_PLOT, OUTPUT_UNCERTAINTY,
    APPLY_SMOOTHING, SMOOTHING_SIGMA,
    STANDARD_LAPSE_RATE,
    INTERPOLATION_REGION, MIN_REGIONAL_STATIONS,
    REGIONAL_BUFFER_KM,
    MODE, NWP_CONFIG, DATA_TIER, RASTER_FILES,
)
from src.fetch_data import fetch_all_data
from src.prepare_features import prepare_station_data, perform_spatial_qc
from src.feature_engineering import engineer_all_features
from src.models import SimpleKrigingBaseline, EnsembleHybridModel
from src.evaluate import evaluate_predictions, print_metrics, compare_models, print_model_comparison
from src.visualize import plot_temperature_map, plot_feature_importance, plot_uncertainty_map, plot_model_comparison, create_comparison_summary_image
from src.export_utils import export_temperature_products
from src.utils import PL_GEOMETRY_2180, nan_gaussian_filter, get_active_geometry, get_region_display_name, is_national_mode, VOIVODESHIP_NAMES, compute_solar_elevation
from src.fetch_nwp import get_nwp_grid, get_nwp_at_stations, get_nwp_cloud_wind_grid, get_nwp_t850_grid

warnings.filterwarnings("ignore")

# Temporal persistence utilities

def _find_previous_output(max_age_hours: float = 2.0):
    """Find the most recent previous temperature output GeoTIFF.
    
    Scans OUTPUT_DIR for timestamped folders containing temperature_2180_*.tif.
    Returns (raster_path, age_hours) or (None, None) if nothing recent enough.
    """
    candidates = []
    for folder in OUTPUT_DIR.iterdir():
        if not folder.is_dir() or folder == RUN_OUTPUT_DIR:
            continue
        try:
            folder_time = datetime.strptime(folder.name, "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            continue
        
        # Find the temperature GeoTIFF in this folder
        tifs = list(folder.glob("temperature_2180_*.tif"))
        if not tifs:
            continue
        
        age_hours = (datetime.now() - folder_time).total_seconds() / 3600
        if age_hours <= max_age_hours:
            candidates.append((tifs[0], age_hours, folder_time))
    
    if not candidates:
        return None, None
    
    # Sort by most recent (smallest age)
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0], candidates[0][1]


def _sample_raster_at_points(raster_path, gdf):
    """Sample a GeoTIFF raster at GeoDataFrame point locations.
    
    Returns 1D array of values (NaN where out-of-bounds or nodata).
    """
    values = np.full(len(gdf), np.nan)
    try:
        gdf_proj = gdf.to_crs(CRS_POLAND) if gdf.crs and str(gdf.crs) != CRS_POLAND else gdf
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            nodata = src.nodata
            for i, geom in enumerate(gdf_proj.geometry):
                try:
                    row, col = rowcol(src.transform, geom.x, geom.y)
                    if 0 <= row < src.height and 0 <= col < src.width:
                        val = float(data[row, col])
                        if nodata is not None and val == nodata:
                            continue
                        if -60 < val < 60:  # sanity range
                            values[i] = val
                except (IndexError, ValueError):
                    continue
    except Exception as e:
        print(f"[Temporal] ⚠️ Could not sample raster: {e}")
    return values


def _add_day_of_year_features(gdf):
    """Add circular day-of-year features for seasonal context.
    
    Returns sin/cos pair encoding day 1-365 as a unit circle position.
    This lets LightGBM learn seasonal modulation of terrain features.
    """
    doy = datetime.now().timetuple().tm_yday
    gdf = gdf.copy()
    gdf['day_of_year_sin'] = np.sin(2 * np.pi * doy / 365.25)
    gdf['day_of_year_cos'] = np.cos(2 * np.pi * doy / 365.25)
    return gdf

# Grid and splut utilities
def create_prediction_grid(resolution=GRID_RESOLUTION):
    """Create empty grid within active region."""
    region_geom = get_active_geometry(buffered=False)
    region_name = INTERPOLATION_REGION if not is_national_mode(INTERPOLATION_REGION) else "Poland"
    print(f"\nCreating {resolution}m Grid for {region_name}...")
    
    bounds = region_geom.bounds
    grid_x_1d = np.arange(bounds[0], bounds[2], resolution)
    grid_y_1d = np.arange(bounds[3], bounds[1], -resolution) # top to bottom
    grid_x, grid_y = np.meshgrid(grid_x_1d, grid_y_1d)
    
    transform = from_bounds(grid_x_1d[0], grid_y_1d[-1], grid_x_1d[-1], grid_y_1d[0], len(grid_x_1d), len(grid_y_1d))
    
    # WGS84 for plotting
    transformer = Transformer.from_crs(CRS_POLAND, CRS_WGS84, always_xy=True)
    grid_lon, grid_lat = transformer.transform(grid_x, grid_y)
    
    # Region mask
    region_mask = rasterize([(region_geom, 1)], out_shape=grid_x.shape, transform=transform, fill=0, dtype='uint8').astype(bool)
    
    # Valid points GDF
    valid_x, valid_y = grid_x[region_mask], grid_y[region_mask]
    grid_gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(valid_x, valid_y)], crs=CRS_POLAND)
    
    print(f"Grid: {grid_x.shape}, Active Points: {len(grid_gdf):,}")
    return grid_gdf, grid_x_1d, grid_y_1d, grid_lon, grid_lat, region_mask

def extract_grid_features_safe(grid_gdf, train_gdf, all_needed_cols, grid_resolution_m=0):
    """Extract features for grid, filling missing with training medians."""
    print("\nExtracting Grid Features...")
    from src.feature_engineering import RasterFeatureExtractor, extract_terrain_features, add_coordinate_features, compute_distance_features, create_feature_interactions

    extractor = RasterFeatureExtractor()

    # Base and Terrain - bilinear interpolation for coarse rasters at sub-km grids
    basic = extractor.extract_all_basic_features(grid_gdf, debug=False,
                                                  grid_resolution_m=grid_resolution_m)
    for c in basic.columns: grid_gdf[c] = basic[c]
    
    # Ensure DEM exists for derivatives
    if 'dem' in grid_gdf.columns:
        grid_gdf['dem'] = grid_gdf['dem'].fillna(train_gdf['dem'].median())

    terrain = extract_terrain_features(grid_gdf, extractor)
    coords = add_coordinate_features(grid_gdf) # gets x_pl, y_pl
    dists = compute_distance_features(grid_gdf, extractor)

    # Release raster memory before NWP interpolation and Kriging
    extractor.release()

    # combine
    for df in [terrain, coords, dists]:
        for c in df.columns: grid_gdf[c] = df[c]
            
    # interactions
    interactions = create_feature_interactions(grid_gdf)
    for c in interactions.columns: grid_gdf[c] = interactions[c]
        
    # Finalize & Impute needed columns
    final_df = pd.DataFrame(index=grid_gdf.index)
    for col in all_needed_cols:
        if col in grid_gdf.columns:
            final_df[col] = grid_gdf[col]
        elif col in train_gdf.columns:
            final_df[col] = train_gdf[col].median()
        else:
            final_df[col] = 0.0
        
        # Last ditch fill
        if final_df[col].isna().any():
            fill_val = train_gdf[col].median() if col in train_gdf else 0
            final_df[col] = final_df[col].fillna(fill_val)
            
    return gpd.GeoDataFrame(final_df, geometry=grid_gdf.geometry, crs=grid_gdf.crs)

def spatial_train_val_test_split(gdf, test_size, val_size, buffer_km):
    """Rigorous 3-way spatial split with buffers."""
    print(f"\nSpatial Split (Buffer: {buffer_km}km)...")
    
    # Helper to remove buffered points
    def remove_buffered(keep_gdf, drop_candidate_gdf, buff_km):
        if buff_km <= 0: return drop_candidate_gdf
        keep_coords = np.array([[g.x, g.y] for g in keep_gdf.to_crs(CRS_POLAND).geometry])
        cand_coords = np.array([[g.x, g.y] for g in drop_candidate_gdf.to_crs(CRS_POLAND).geometry])
        tree = cKDTree(keep_coords)
        dists, _ = tree.query(cand_coords)
        mask = dists >= (buff_km * 1000)
        return drop_candidate_gdf[mask].copy()

    # Split Test
    train_val_raw, test = train_test_split(gdf, test_size=test_size, random_state=RANDOM_STATE)
    train_val_buffered = remove_buffered(test, train_val_raw, buffer_km)
    
    # Split Val from remaining
    val_adj_size = val_size / (1.0 - test_size)
    train_raw, val = train_test_split(train_val_buffered, test_size=val_adj_size, random_state=RANDOM_STATE)
    train = remove_buffered(val, train_raw, buffer_km)
    
    print(f"Split: Train={len(train)}, Val={len(val)}, Test={len(test)} (Buffered: {len(gdf)-len(train)-len(val)-len(test)})")
    return train, val, test

# Core pipeline
def main():
    print("-"*60 + "\nHRMTA: Model start\n" + "-"*60)

    # Configuration summary
    _tier_labels = {"lite": "Lite (DEM only)", "standard": "Standard (DEM + terrain physics)", "full": "Full (all rasters)"}
    print(f"   Mode: {MODE.capitalize()} | Data: {_tier_labels.get(DATA_TIER, DATA_TIER)} | Grid: {GRID_RESOLUTION}m")
    print(f"   Rasters: {len(RASTER_FILES)} layers ({', '.join(RASTER_FILES.keys())})")

    # Data & Features
    raw_data = fetch_all_data()
    data_fetch_time = datetime.now(timezone.utc)  # Capture fetch timestamp
    if len(raw_data) < 50: return
    
    stations_gdf = prepare_station_data(raw_data)

    # Regional filtering
    if not is_national_mode(INTERPOLATION_REGION):
        buffered_geom = get_active_geometry(buffered=True)
        stations_gdf = stations_gdf.to_crs(CRS_POLAND)
        regional_mask = stations_gdf.geometry.within(buffered_geom)
        regional_count = regional_mask.sum()
        
        print(f"\nRegional mode: {INTERPOLATION_REGION}")
        print(f"   Stations in region + {REGIONAL_BUFFER_KM}km buffer: {regional_count}")
        
        if regional_count < MIN_REGIONAL_STATIONS:
            print(f"   ⚠️ Only {regional_count} stations (< {MIN_REGIONAL_STATIONS}). Results may be unreliable.")
        
        stations_gdf = stations_gdf[regional_mask].copy()
        
        # Filter out IMGW stations with provName that doesn't match the region
        canonical_region = VOIVODESHIP_NAMES.get(INTERPOLATION_REGION.lower(), INTERPOLATION_REGION.lower())

        if 'provName' in stations_gdf.columns:
            unbuffered_geom = get_active_geometry(buffered=False)
            inside_region = stations_gdf.geometry.within(unbuffered_geom)
            imgw_mask = stations_gdf['source'] == 'IMGW'
            provname_mismatch = inside_region & imgw_mask & (stations_gdf['provName'].str.lower() != canonical_region)
            mismatch_count = provname_mismatch.sum()
            
            if mismatch_count > 0:
                mismatched = stations_gdf[provname_mismatch]['station'].tolist()[:5]
                print(f"   ⚠️ Filtering {mismatch_count} misplaced IMGW stations: {mismatched}")
                stations_gdf = stations_gdf[~provname_mismatch].copy()
        
        stations_gdf = stations_gdf.to_crs(CRS_WGS84)

    # Engineer features
    stations_gdf, all_eng_cols = engineer_all_features(stations_gdf)
    
    nwp_enabled = False
    _nwp_target_time = None
    if MODE == "pro":
        print("\n--- Pro Mode: NWP Feature Integration ---")
        # Temporal lock
        _nwp_target_time = datetime.now(timezone.utc)
        stations_gdf = get_nwp_at_stations(stations_gdf, target_time=_nwp_target_time)
        
        if 'nwp_temp' in stations_gdf.columns and stations_gdf['nwp_temp'].notna().sum() > 50:
            nwp_enabled = True
            
            # NWP temperature as a direct feature
            stations_gdf['nwp_t2m'] = stations_gdf['nwp_temp'].copy()
            
            # Compute NWP local error
            print("[Pro] Computing NWP local error field (trusted-only LOO-IDW)...")
            stations_proj = stations_gdf.to_crs(CRS_POLAND)
            coords = np.array([[g.x, g.y] for g in stations_proj.geometry])

            # Identify trusted stations for error baseline
            _trusted_sources = {'IMGW', 'EDWIN'}
            _is_trusted = stations_gdf['source'].isin(_trusted_sources).values
            _trusted_idx = np.where(_is_trusted)[0]
            _n_trusted_nwp = len(_trusted_idx)

            # Compute errors from ALL stations (needed for per-station self-error)
            _all_signed_errors = stations_gdf['temp'].values - stations_gdf['nwp_temp'].values

            # Trusted-only error arrays (clean NWP quality signal)
            _trusted_signed_errors = _all_signed_errors[_trusted_idx]
            _trusted_abs_errors = np.abs(_trusted_signed_errors)

            # Domain-mean signed bias from trusted stations only
            domain_mean_bias = np.nanmean(_trusted_signed_errors)
            _trusted_debiased_abs_errors = np.abs(_trusted_signed_errors - domain_mean_bias)

            # Also keep all-station versions for grid interpolation fallback
            nwp_abs_errors = np.abs(_all_signed_errors)
            nwp_signed_errors = _all_signed_errors.copy()
            nwp_debiased_abs_errors = np.abs(_all_signed_errors - domain_mean_bias)

            # Build kd-tree from TRUSTED stations only
            _trusted_coords = coords[_trusted_idx]
            trusted_tree = cKDTree(_trusted_coords)

            # Also keep all-station tree for grid interpolation
            station_tree = cKDTree(coords)

            # Query: for each station, find nearest trusted neighbors
            K_NEIGHBORS = 25
            MAX_RADIUS_M = 150000.0  # 150km (wider for sparser network)
            dists, idxs = trusted_tree.query(coords, k=min(K_NEIGHBORS, _n_trusted_nwp))

            print(f"[Pro] Trusted stations for NWP baseline: {_n_trusted_nwp} "
                  f"(domain bias: {domain_mean_bias:+.2f}°C)")
            
            nwp_local_error = np.zeros(len(coords))
            nwp_local_bias = np.zeros(len(coords))
            nwp_debiased_error = np.zeros(len(coords))
            
            # Regime detection
            if 'nwp_cloud' in stations_gdf.columns and 'nwp_wind' in stations_gdf.columns:
                solar = compute_solar_elevation(stations_gdf, _nwp_target_time)
                ccn = ((stations_gdf['nwp_cloud'].fillna(0.5) < 0.3) & 
                       (stations_gdf['nwp_wind'].fillna(3.0) < 3.0) & 
                       (solar < -6.0)).astype(float)
                ccn_frac = float(np.clip(np.nanmean(ccn), 0.0, 1.0))
            else:
                ccn_frac = 0.0
                
            terrain_alpha = float(np.clip(ccn_frac / 0.5, 0.0, 1.0))
            
            # Hyperparameters
            L_NWP = 20000.0  # 20km characteristic length
            ELEV_SCALE = 120.0
            TPI_SCALE = 1.2
            TERRAIN_FLOOR = 0.05
            
            station_elev = stations_gdf.get('dem', pd.Series(0, index=stations_gdf.index)).values
            station_tpi = stations_gdf.get('tpi_2000', pd.Series(0, index=stations_gdf.index)).values

            # Terrain features at trusted stations (for terrain weighting)
            _trusted_elev = station_elev[_trusted_idx]
            _trusted_tpi = station_tpi[_trusted_idx]

            for i in range(len(coords)):
                if _is_trusted[i]:
                    neighbor_dists = dists[i, 1:]
                    neighbor_tidxs = idxs[i, 1:]  # indices into trusted arrays
                else:
                    neighbor_dists = dists[i, :]
                    neighbor_tidxs = idxs[i, :]

                # Only use neighbors within radius
                within = neighbor_dists < MAX_RADIUS_M
                if within.sum() == 0:
                    nwp_local_error[i] = np.mean(_trusted_abs_errors)
                    nwp_local_bias[i] = np.mean(_trusted_signed_errors)
                    nwp_debiased_error[i] = np.mean(_trusted_debiased_abs_errors)
                    continue

                d = neighbor_dists[within]
                tidxs = neighbor_tidxs[within]  # indices into trusted arrays

                # Gaussian geographic weight (matches grid kernel)
                w_geo = np.exp(-(d / L_NWP) ** 2)

                # terrain weighting (using trusted stations' terrain)
                elev_diff = np.abs(station_elev[i] - _trusted_elev[tidxs])
                tpi_diff = np.abs(station_tpi[i] - _trusted_tpi[tidxs])

                raw_sim = np.exp(-elev_diff / ELEV_SCALE - tpi_diff / TPI_SCALE)
                raw_sim = TERRAIN_FLOOR + (1.0 - TERRAIN_FLOOR) * raw_sim
                terrain_sim = (1.0 - terrain_alpha) + terrain_alpha * raw_sim

                w = w_geo * terrain_sim

                nwp_local_error[i] = np.average(_trusted_abs_errors[tidxs], weights=w)
                nwp_local_bias[i] = np.average(_trusted_signed_errors[tidxs], weights=w)
                nwp_debiased_error[i] = np.average(_trusted_debiased_abs_errors[tidxs], weights=w)
            
            stations_gdf['nwp_local_error'] = nwp_local_error
            stations_gdf['nwp_signed_bias'] = nwp_local_bias
            stations_gdf['nwp_debiased_error'] = nwp_debiased_error
            
            # Physics-activated interaction features
            if 'nwp_cloud' in stations_gdf.columns and 'nwp_wind' in stations_gdf.columns:
                cloud = stations_gdf['nwp_cloud'].fillna(0.5).values
                wind = stations_gdf['nwp_wind'].fillna(3.0).values
                _solar = compute_solar_elevation(stations_gdf, data_fetch_time)
                _night_factor = np.clip(-_solar / 6.0, 0.0, 1.0)
                decoupling = (1.0 - np.clip(cloud, 0, 1)) * np.exp(-wind / 2.0) * _night_factor
                stations_gdf['decoupling_index'] = decoupling
                
                # Physics-activated terrain features (if terrain rasters available)
                if 'svf' in stations_gdf.columns:
                    stations_gdf['radiation_loss'] = stations_gdf['svf'].fillna(0).values * decoupling
                if 'cap_anomaly' in stations_gdf.columns:
                    stations_gdf['cold_pool_activation'] = (-stations_gdf['cap_anomaly'].fillna(0).values) * decoupling
                if 'hand' in stations_gdf.columns:
                    hand_vals = stations_gdf['hand'].fillna(50.0).values
                    stations_gdf['hand_cold_pool'] = np.exp(-hand_vals / 30.0) * decoupling
                if 'tpi_2000' in stations_gdf.columns:
                    stations_gdf['wind_exposure'] = stations_gdf['tpi_2000'].fillna(0).values * wind
                if 'canopy_height' in stations_gdf.columns:
                    canopy_h = stations_gdf['canopy_height'].fillna(0).values
                    stations_gdf['canopy_trapping'] = np.log1p(canopy_h) * decoupling
            
            # Print summary
            nwp_err_mean = nwp_abs_errors.mean()
            nwp_err_std = nwp_abs_errors.std()
            print(f"[Pro] NWP at stations: {stations_gdf['nwp_temp'].notna().sum()} available")
            print(f"[Pro] NWP absolute error: mean={nwp_err_mean:.2f}°C, std={nwp_err_std:.2f}°C")
            print(f"[Pro] NWP local error (LOO): mean={nwp_local_error.mean():.2f}°C, std={nwp_local_error.std():.2f}°C")
            print(f"[Pro] NWP signed bias (LOO): mean={nwp_local_bias.mean():+.2f}°C (+ = NWP cold, - = NWP warm)")
            print(f"[Pro] NWP domain-mean bias: {domain_mean_bias:+.2f}°C")
            print(f"[Pro] NWP debiased error (LOO): mean={nwp_debiased_error.mean():.2f}°C")
            if 'decoupling_index' in stations_gdf.columns:
                print(f"[Pro] Decoupling index: mean={stations_gdf['decoupling_index'].mean():.3f}")
            
            # NWP Trust Weight
            NWP_TRUST_SIGMA = 1.0
            stations_gdf['nwp_trust'] = np.exp(
                -(nwp_debiased_error ** 2) / (2 * NWP_TRUST_SIGMA ** 2)
            )
            print(f"[Pro] NWP trust (debiased): mean={stations_gdf['nwp_trust'].mean():.3f}")
            
            print(f"[Pro] Training on absolute temperature with NWP features")
        else:
            print("[Pro] ⚠️ Insufficient NWP data at stations, falling back to Standard mode")
    
    # NWP cloud-derived features
    if nwp_enabled and 'nwp_cloud' in stations_gdf.columns and 'nwp_wind' in stations_gdf.columns:
        solar_elev = compute_solar_elevation(stations_gdf, data_fetch_time)
        cloud_col = stations_gdf['nwp_cloud'].fillna(0.5)
        wind_col = stations_gdf['nwp_wind'].fillna(3.0)
        
        stations_gdf['calm_clear_night'] = (
            (cloud_col < 0.3) &
            (wind_col < 3.0) &
            (solar_elev < -6.0)
        ).astype(float)
        
        night_count = int(stations_gdf['calm_clear_night'].sum())
        print(f"[Pro] calm_clear_night: {night_count}/{len(stations_gdf)} stations")
        print(f"[Pro] Solar elevation: {solar_elev.min():.1f}° to {solar_elev.max():.1f}°")
        
    # NWP integration
    if nwp_enabled:
        print(f"\n--- Pro Mode: Enhanced NWP Features ---")
        
        # nwp_t2m_anomaly: NWP temp minus terrain-predicted temp
        from sklearn.linear_model import HuberRegressor as _Huber
        _trend_data = stations_gdf[TREND_FEATURES].fillna(0).values
        _quick_huber = _Huber(epsilon=1.35, max_iter=200)
        _quick_huber.fit(_trend_data, stations_gdf['temp'].values)
        _trend_pred = _quick_huber.predict(_trend_data)
        stations_gdf['nwp_t2m_anomaly'] = stations_gdf['nwp_t2m'].values - _trend_pred
        print(f"[Pro v2] nwp_t2m_anomaly: mean={stations_gdf['nwp_t2m_anomaly'].mean():+.2f}°C")
        
        # nwp_regime_stability: spatial coherence of NWP bias
        _stab_dists, _stab_idxs = station_tree.query(coords, k=K_NEIGHBORS + 1)
        nwp_regime_stab = np.zeros(len(coords))
        for i in range(len(coords)):
            neighbor_dists_v = _stab_dists[i, 1:]
            neighbor_idxs_v = _stab_idxs[i, 1:]
            within_v = neighbor_dists_v < MAX_RADIUS_M
            if within_v.sum() > 2:
                nwp_regime_stab[i] = np.std(nwp_signed_errors[neighbor_idxs_v[within_v]])
            else:
                nwp_regime_stab[i] = np.std(nwp_signed_errors)
        stations_gdf['nwp_regime_stability'] = nwp_regime_stab
        print(f"[Pro v2] nwp_regime_stability: mean={nwp_regime_stab.mean():.2f}°C")
        
        # ICON-EU integration
        try:
            from src.fetch_icon import get_icon_at_stations
            print("[Pro v2] Fetching ICON-EU data...")
            stations_gdf = get_icon_at_stations(stations_gdf, target_time=_nwp_target_time)
            
            icon_available = (
                'icon_t2m' in stations_gdf.columns and
                stations_gdf['icon_t2m'].notna().sum() > 50
            )
            
            if icon_available:
                stations_gdf['nwp_model_agreement'] = np.abs(
                    stations_gdf['nwp_t2m'].values - stations_gdf['icon_t2m'].values
                )
                disagree_mean = stations_gdf['nwp_model_agreement'].mean()
                print(f"[Pro v2] nwp_model_agreement: mean={disagree_mean:.2f}°C")
                
                if 'icon_hsurf' in stations_gdf.columns and 'dem' in stations_gdf.columns:
                    stations_gdf['nwp_elev_mismatch'] = np.abs(
                        stations_gdf['dem'].values - stations_gdf['icon_hsurf'].values
                    )
                    mismatch_mean = stations_gdf['nwp_elev_mismatch'].mean()
                    print(f"[Pro v2] nwp_elev_mismatch: mean={mismatch_mean:.0f}m")
            else:
                print("[Pro v2] ⚠ ICON-EU data insufficient, features set to NaN")
                for col in ['nwp_model_agreement', 'nwp_elev_mismatch']:
                    if col not in stations_gdf.columns:
                        stations_gdf[col] = np.nan
        except Exception as e:
            print(f"[Pro v2] ⚠ ICON-EU fetch failed: {e}")
            for col in ['icon_t2m', 'icon_cloud', 'icon_wind', 'icon_t850',
                        'nwp_model_agreement', 'nwp_elev_mismatch']:
                if col not in stations_gdf.columns:
                    stations_gdf[col] = np.nan

        # Inversion strength (T850 - T2m)
        _t850_src = None
        if 'nwp_t850' in stations_gdf.columns and stations_gdf['nwp_t850'].notna().sum() > 50:
            stations_gdf['inversion_strength'] = (
                stations_gdf['nwp_t850'].values - stations_gdf['nwp_t2m'].values
            )
            _t850_src = 'HARMONIE'
        elif 'icon_t850' in stations_gdf.columns and stations_gdf['icon_t850'].notna().sum() > 50:
            stations_gdf['inversion_strength'] = (
                stations_gdf['icon_t850'].values - stations_gdf['icon_t2m'].values
            )
            _t850_src = 'ICON-EU'

        if _t850_src is not None:
            inv = stations_gdf['inversion_strength']
            n_inv = int((inv > 0).sum())
            print(f"[Pro v2] inversion_strength ({_t850_src}): "
                  f"mean={inv.mean():+.1f}°C, {n_inv}/{len(inv)} stations inverted")
        else:
            stations_gdf['inversion_strength'] = np.nan
            print("[Pro v2] ⚠ No T850 available, inversion_strength = NaN")

        # Summary of v2 features
        v2_features = ['nwp_t2m_anomaly', 'nwp_regime_stability', 'nwp_model_agreement',
                       'nwp_elev_mismatch', 'icon_t2m', 'icon_cloud', 'icon_wind',
                       'inversion_strength']
        available_v2 = [f for f in v2_features if f in stations_gdf.columns and stations_gdf[f].notna().sum() > 30]
        print(f"[Pro v2] Added {len(available_v2)} v2 features: {available_v2}")

    # Pre-QC NWP Domain Debiasing
    _qc_nwp_original = None
    if nwp_enabled and 'nwp_temp' in stations_gdf.columns and 'nwp_signed_bias' in stations_gdf.columns:
        local_bias = stations_gdf['nwp_signed_bias'].fillna(0).values
        domain_bias = float(np.nanmedian(local_bias))
        max_local = float(np.nanmax(np.abs(local_bias)))
        
        # Apply if domain bias is significant OR local bias varies strongly
        if abs(domain_bias) > 0.3 or max_local > 1.0:
            # Save original as pandas Series
            _qc_nwp_original = stations_gdf['nwp_temp'].copy()  # Series
            _qc_nwp_t2m_original = stations_gdf['nwp_t2m'].copy() if 'nwp_t2m' in stations_gdf.columns else None
            
            # Apply LOCAL bias correction
            stations_gdf['nwp_temp'] = stations_gdf['nwp_temp'] + local_bias
            if 'nwp_t2m' in stations_gdf.columns:
                stations_gdf['nwp_t2m'] = stations_gdf['nwp_t2m'] + local_bias
            
            n_corrected = int(np.sum(np.abs(local_bias) > 0.3))
            print(f"\n[QC Debias v1.4.0] Local NWP correction before QC:")
            print(f"   Domain bias: {domain_bias:+.2f}°C, max local: {max_local:.2f}°C")
            print(f"   {n_corrected}/{len(local_bias)} stations shifted >0.3°C")
        else:
            print(f"\n[QC Debias] Bias negligible (domain: {domain_bias:+.2f}°C, "
                  f"max local: {max_local:.2f}°C) - skipped")
    
    # Spatial QC - FS-ISCT
    stations_gdf = perform_spatial_qc(stations_gdf)

    # Restore original NWP values
    if _qc_nwp_original is not None:
        stations_gdf['nwp_temp'] = _qc_nwp_original
        if _qc_nwp_t2m_original is not None and 'nwp_t2m' in stations_gdf.columns:
            stations_gdf['nwp_t2m'] = _qc_nwp_t2m_original
        print(f"[QC Debias] NWP restored to original (local correction was QC-only)")
    
    # Consumer station bias correction
    TRUSTED_SOURCES = {'IMGW', 'EDWIN'}
    CONSUMER_SOURCES = {'NETATMO'}
    
    if 'source' in stations_gdf.columns:
        trusted_mask = stations_gdf['source'].isin(TRUSTED_SOURCES)
        consumer_mask = stations_gdf['source'].isin(CONSUMER_SOURCES)
        n_trusted = trusted_mask.sum()
        n_consumer = consumer_mask.sum()

        if n_trusted >= 50 and n_consumer >= 50:
            print(f"\n[Bias Correction] Anchoring {n_consumer} consumer stations to {n_trusted} trusted stations...")
            
            # Project to meters for distance computation
            bc_proj = stations_gdf.to_crs(CRS_POLAND)
            trusted_coords = np.column_stack([
                bc_proj.loc[trusted_mask].geometry.x.values,
                bc_proj.loc[trusted_mask].geometry.y.values
            ])
            cons_coords = np.column_stack([
                bc_proj.loc[consumer_mask].geometry.x.values,
                bc_proj.loc[consumer_mask].geometry.y.values
            ])
            trusted_temps = stations_gdf.loc[trusted_mask, 'temp'].values
            cons_temps = stations_gdf.loc[consumer_mask, 'temp'].values

            # Elevation for lapse-rate adjustment
            trusted_elev = stations_gdf.loc[trusted_mask, 'dem'].values if 'dem' in stations_gdf.columns else None
            cons_elev = stations_gdf.loc[consumer_mask, 'dem'].values if 'dem' in stations_gdf.columns else None

            # Dynamic lapse rate from the fitted Huber (or standard fallback)
            _bias_lapse = STANDARD_LAPSE_RATE

            # Find nearest trusted stations for each consumer station
            trusted_tree = cKDTree(trusted_coords)
            K_BIAS = 7          # neighbors to consider
            MAX_BIAS_DIST = 50000.0  # 50km radius
            MIN_NEIGHBORS = 3   # minimum for robust median

            bias_dists, bias_idxs = trusted_tree.query(cons_coords, k=K_BIAS)

            # QC confidence for gating bias correction
            cons_qc = (stations_gdf.loc[consumer_mask, 'qc_confidence'].values
                      if 'qc_confidence' in stations_gdf.columns
                      else np.ones(n_consumer))

            corrections = np.zeros(n_consumer)
            corrected_count = 0
            attenuated_count = 0
            skipped_low_qc = 0

            for i in range(n_consumer):
                if cons_qc[i] < 0.15:
                    skipped_low_qc += 1
                    continue

                within = bias_dists[i] < MAX_BIAS_DIST
                if within.sum() < MIN_NEIGHBORS:
                    continue

                neighbor_idx = bias_idxs[i][within]
                neighbor_temps = trusted_temps[neighbor_idx]

                # Elevation-adjusted comparison
                if trusted_elev is not None and cons_elev is not None:
                    elev_diff = trusted_elev[neighbor_idx] - cons_elev[i]
                    valid_elev = np.isfinite(elev_diff)
                    if valid_elev.sum() >= MIN_NEIGHBORS:
                        neighbor_temps_adj = neighbor_temps[valid_elev] + _bias_lapse * elev_diff[valid_elev]
                    else:
                        neighbor_temps_adj = neighbor_temps
                else:
                    neighbor_temps_adj = neighbor_temps

                local_offset = cons_temps[i] - np.median(neighbor_temps_adj)

                if abs(local_offset) > 0.2 and abs(local_offset) < 5.0:
                    correction = -local_offset
                    if cons_qc[i] < 0.5:
                        correction *= cons_qc[i] / 0.5
                        attenuated_count += 1
                    corrections[i] = correction
                    corrected_count += 1

            # Apply corrections
            cons_idx = stations_gdf.index[consumer_mask]
            stations_gdf.loc[cons_idx, 'temp'] = cons_temps + corrections

            applied = corrections[corrections != 0]
            if len(applied) > 0:
                print(f"[Bias Correction] Corrected {corrected_count}/{n_consumer} consumer stations")
                if skipped_low_qc > 0 or attenuated_count > 0:
                    print(f"[Bias Correction] QC gating: {skipped_low_qc} skipped (qc<0.15), "
                          f"{attenuated_count} attenuated (qc<0.5)")
                print(f"[Bias Correction] Mean correction: {applied.mean():+.2f}°C "
                      f"(median: {np.median(applied):+.2f}°C, "
                      f"std: {applied.std():.2f}°C)")
                print(f"[Bias Correction] Direction: {(applied > 0).sum()} warmed, "
                      f"{(applied < 0).sum()} cooled")
            else:
                print("[Bias Correction] No significant biases detected - no corrections applied")
        else:
            print(f"\n[Bias Correction] Skipped (trusted: {n_trusted}, consumer: {n_consumer} - insufficient)")
    
    # Rebuild station tree and NWP error arrays after QC + bias correction
    if nwp_enabled and 'nwp_temp' in stations_gdf.columns:
        stations_proj = stations_gdf.to_crs(CRS_POLAND)
        coords = np.array([[g.x, g.y] for g in stations_proj.geometry])
        station_tree = cKDTree(coords)
        nwp_abs_errors = np.abs(stations_gdf['temp'].values - stations_gdf['nwp_temp'].values)
        nwp_signed_errors = stations_gdf['temp'].values - stations_gdf['nwp_temp'].values
    
    # Temporal persistence features
    prev_raster, prev_age = _find_previous_output(max_age_hours=2.0)
    if prev_raster is not None:
        prev_temps = _sample_raster_at_points(prev_raster, stations_gdf)
        valid_prev = np.isfinite(prev_temps).sum()
        stations_gdf['t_prev'] = prev_temps
        print(f"\n[Temporal] Previous output: {prev_raster.parent.name} ({prev_age:.1f}h ago)")
        print(f"[Temporal] Sampled at {valid_prev}/{len(stations_gdf)} stations")
        
        if 'nwp_t2m' in stations_gdf.columns:
            t_prev_vals = stations_gdf['t_prev'].values
            nwp_vals = stations_gdf['nwp_t2m'].values
            divergence = np.abs(t_prev_vals - nwp_vals)
            stale_mask = divergence > 3.0
            n_stale = np.nansum(stale_mask)
            if n_stale > 0:
                stations_gdf.loc[stale_mask, 't_prev'] = np.nan
                print(f"[Temporal] Invalidated {int(n_stale)} stale t_prev values "
                      f"(|t_prev - NWP| > 3°C)")
    else:
        stations_gdf['t_prev'] = np.nan
        print(f"\n[Temporal] No recent output found (within 2h). Features will be NaN.")
    
    # Day-of-year circular features
    stations_gdf = _add_day_of_year_features(stations_gdf)
    
    # Feature organization
    trend_cols = TREND_FEATURES
    
    # Everything else engineered (exclude non-feature columns)
    non_features = set(['station', 'temp', 'geometry', 'source', 'lat', 'lon', 'nwp_temp', 'qc_confidence'] + trend_cols)
    env_cols = [c for c in all_eng_cols if c not in non_features]
    
    # Add temporal and seasonal features (computed after engineer_all_features)
    for extra_feat in ['t_prev', 'day_of_year_sin', 'day_of_year_cos']:
        if extra_feat in stations_gdf.columns and extra_feat not in env_cols:
            env_cols.append(extra_feat)
    
    # In Pro mode, add NWP-derived features to LightGBM
    if nwp_enabled:
        pro_features = []
        pro_candidates = [
            'nwp_t2m', 'nwp_cloud', 'nwp_wind', 'nwp_local_error', 'nwp_signed_bias',
            'nwp_debiased_error',
            'decoupling_index', 'radiation_loss', 'cold_pool_activation', 'wind_exposure',
            'canopy_trapping',
            'hand_cold_pool',
            'calm_clear_night',
        ]
        
        # Multi-model and ICON-EU features
        pro_candidates.extend([
            'nwp_t2m_anomaly', 'nwp_regime_stability',
            'nwp_model_agreement', 'nwp_elev_mismatch',
            'icon_t2m', 'icon_cloud', 'icon_wind',
            'inversion_strength',
        ])
        
        for feat in pro_candidates:
            if feat in stations_gdf.columns and stations_gdf[feat].notna().sum() > 30:
                pro_features.append(feat)
        
        env_cols = env_cols + pro_features
        print(f"[Pro] Added {len(pro_features)} features: {pro_features}")
    
    print(f"\nFeatures: Trend={trend_cols}, EnvML={len(env_cols)} features")
    
    # Splitting + Training
    train_gdf, val_gdf, test_gdf = spatial_train_val_test_split(stations_gdf, TEST_SIZE, VAL_SIZE, SPATIAL_BUFFER_KM)
    
    # Skip inline benchmark if too few training stations (common in regional mode)
    MIN_TRAIN_FOR_BENCHMARK = 50
    run_benchmark = len(train_gdf) >= MIN_TRAIN_FOR_BENCHMARK
    
    if run_benchmark:
        # Baseline
        print("\n--- Training Baseline (Kriging) ---")
        baseline = SimpleKrigingBaseline().fit(train_gdf)
        base_test_pred = baseline.predict(test_gdf)
        base_metrics = evaluate_predictions(test_gdf['temp'], base_test_pred)
        print_metrics(base_metrics, "Baseline Test Results")
        
        # Ensemble
        print("\n--- Training Ensemble (Robust Stacking) ---")
        kriging_scale = 0.7
        active_lgbm_params = LIGHTGBM_PARAMS
        model = EnsembleHybridModel(trend_cols, env_cols, ENSEMBLE_N_MODELS, ENSEMBLE_SEEDS, 
                                     lgbm_params=active_lgbm_params, kriging_scale=kriging_scale)
        model.fit(train_gdf)
        
        # Evaluation
        print("\n--- Evaluation ---")
        # val
        val_pred, val_unc = model.predict_with_uncertainty(val_gdf)
        print_metrics(evaluate_predictions(val_gdf['temp'], val_pred), "Validation Set")
        
        # Test
        test_pred, test_unc = model.predict_with_uncertainty(test_gdf)
        test_metrics = evaluate_predictions(test_gdf['temp'], test_pred)
        print_metrics(test_metrics, "FINAL TEST RESULTS")
        
        # Comparison
        comp_df = compare_models(test_gdf['temp'], base_test_pred, test_pred, ("Kriging", "Hybrid"))
        print_model_comparison(comp_df)
    else:
        print(f"\n⚠️  Skipping inline benchmark: only {len(train_gdf)} training stations after spatial buffer")
        print(f"    (minimum {MIN_TRAIN_FOR_BENCHMARK} needed for meaningful evaluation)")
        print(f"    Use benchmark_sloocv.py for proper evaluation, or run in Poland-wide mode.")
        test_metrics = None
        comp_df = None
    
    # Final fit on ALL data (produces the actual output map)
    print("\n--- Final Training ---")
    kriging_scale = 0.7
    active_lgbm_params = LIGHTGBM_PARAMS
    model = EnsembleHybridModel(trend_cols, env_cols, ENSEMBLE_N_MODELS, ENSEMBLE_SEEDS,
                                 lgbm_params=active_lgbm_params, kriging_scale=kriging_scale)
    model.fit(stations_gdf)
    
    # Importance
    imp_df = model.get_feature_importance()
    print(f"\nTop Features:\n{imp_df.head(10).to_string(index=False)}")

    # Grid Prediction and export
    print("\n--- Grid Prediction ---")
    grid_raw, gx, gy, glon, glat, mask = create_prediction_grid(GRID_RESOLUTION)
    grid_ready = extract_grid_features_safe(grid_raw, stations_gdf, trend_cols + env_cols,
                                             grid_resolution_m=GRID_RESOLUTION)
    
    # Grid-level NWP feature extraction
    if nwp_enabled:
        print("\n[Pro] Extracting NWP features for grid...")
        nwp_grid = get_nwp_grid(gx, gy, target_time=_nwp_target_time)

        if nwp_grid is not None:
            nwp_flat = nwp_grid[mask]
            valid_nwp = ~np.isnan(nwp_flat)
            
            # NWP temperature as grid feature
            grid_ready['nwp_t2m'] = nwp_flat
            print(f"[Pro] NWP grid: {valid_nwp.sum()}/{len(nwp_flat)} valid points")
            
            # Compute NWP local error at grid points using trusted stations only
            grid_coords_proj = np.column_stack([grid_ready.geometry.x.values, grid_ready.geometry.y.values])
            K_GRID = min(30, _n_trusted_nwp)  # cap at available trusted stations
            grid_dists, grid_idxs = trusted_tree.query(grid_coords_proj, k=K_GRID)

            # Terrain-weighted gaussian kernel

            # Regime detection
            if 'calm_clear_night' in stations_gdf.columns:
                ccn_frac = float(np.clip(np.nanmean(stations_gdf['calm_clear_night']), 0.0, 1.0))
            else:
                ccn_frac = 0.0
            terrain_alpha = float(np.clip(ccn_frac / 0.5, 0.0, 1.0))

            # Gaussian kernel with 20km characteristic length
            L_NWP = 20000.0  # 20km smoothing radius (meters, EPSG:2180)
            ELEV_SCALE = 120.0
            TPI_SCALE = 1.2
            TERRAIN_FLOOR = 0.05

            # Gaussian spatial weights
            d = grid_dists
            w_geo = np.exp(-(d / L_NWP) ** 2)

            grid_elev = grid_ready.get('dem', pd.Series(0, index=grid_ready.index)).fillna(0).values
            grid_tpi = grid_ready.get('tpi_2000', pd.Series(0, index=grid_ready.index)).fillna(0).values

            # grid_idxs indexes into trusted arrays (not full station arrays)
            elev_diff = np.abs(grid_elev[:, None] - _trusted_elev[grid_idxs])
            tpi_diff = np.abs(grid_tpi[:, None] - _trusted_tpi[grid_idxs])

            raw_sim = np.exp(-elev_diff / ELEV_SCALE - tpi_diff / TPI_SCALE)
            raw_sim = TERRAIN_FLOOR + (1.0 - TERRAIN_FLOOR) * raw_sim

            terrain_sim = (1.0 - terrain_alpha) + terrain_alpha * raw_sim

            w = w_geo * terrain_sim
            w_sum = w.sum(axis=1, keepdims=True)

            grid_nwp_error = (w * _trusted_abs_errors[grid_idxs]).sum(axis=1) / w_sum.ravel()
            grid_nwp_bias = (w * _trusted_signed_errors[grid_idxs]).sum(axis=1) / w_sum.ravel()
            grid_nwp_debiased_error = (w * _trusted_debiased_abs_errors[grid_idxs]).sum(axis=1) / w_sum.ravel()
            
            grid_ready['nwp_local_error'] = grid_nwp_error
            grid_ready['nwp_signed_bias'] = grid_nwp_bias
            grid_ready['nwp_debiased_error'] = grid_nwp_debiased_error
            print(f"[Pro] Grid NWP signed bias: mean={grid_nwp_bias.mean():+.2f}°C")
            
            # Grid NWP trust weight, uses debiased error (same as stations)
            NWP_TRUST_SIGMA = 1.0
            grid_nwp_trust = np.exp(-(grid_nwp_debiased_error ** 2) / (2 * NWP_TRUST_SIGMA ** 2))
            grid_ready['nwp_trust'] = grid_nwp_trust
            print(f"[Pro] Grid NWP trust (debiased): mean={grid_nwp_trust.mean():.3f}")
            
            # NWP cloud and wind from GRIB directly (spatially varying)
            _cw_grid = get_nwp_cloud_wind_grid(gx, gy, target_time=_nwp_target_time)
            if _cw_grid is not None:
                if 'cloud_cover' in _cw_grid:
                    cloud_g = _cw_grid['cloud_cover'][mask]
                    cloud_g = np.clip(np.nan_to_num(cloud_g, nan=0.5), 0.0, 1.0)
                    grid_ready['nwp_cloud'] = cloud_g
                    print(f"[Pro] Grid cloud from GRIB: mean={cloud_g.mean():.2f}")
                else:
                    cloud_g = np.full(len(grid_ready), 0.5)
                    grid_ready['nwp_cloud'] = cloud_g
                if 'wind_speed' in _cw_grid:
                    wind_g = _cw_grid['wind_speed'][mask]
                    wind_g = np.clip(np.nan_to_num(wind_g, nan=3.0), 0.0, 50.0)
                    grid_ready['nwp_wind'] = wind_g
                    print(f"[Pro] Grid wind from GRIB: mean={wind_g.mean():.1f} m/s")
                else:
                    wind_g = np.full(len(grid_ready), 3.0)
                    grid_ready['nwp_wind'] = wind_g
            else:
                # Fallback: use training median (degrades to v1.4.0 behavior)
                cloud_g = np.full(len(grid_ready), 0.5)
                wind_g = np.full(len(grid_ready), 3.0)
                grid_ready['nwp_cloud'] = cloud_g
                grid_ready['nwp_wind'] = wind_g
                print("[Pro] Grid cloud/wind: using fallback constants (GRIB unavailable)")
            
            # Physics interaction features for grid (same formulas as training)
            _grid_solar = compute_solar_elevation(grid_ready, data_fetch_time)
            _grid_night_factor = np.clip(-_grid_solar / 6.0, 0.0, 1.0)
            decoupling_g = (1.0 - np.clip(cloud_g, 0, 1)) * np.exp(-wind_g / 2.0) * _grid_night_factor
            grid_ready['decoupling_index'] = decoupling_g
            
            if 'svf' in grid_ready.columns:
                grid_ready['radiation_loss'] = grid_ready['svf'].fillna(0).values * decoupling_g
            if 'cap_anomaly' in grid_ready.columns:
                grid_ready['cold_pool_activation'] = (-grid_ready['cap_anomaly'].fillna(0).values) * decoupling_g
            if 'hand' in grid_ready.columns:
                hand_g = grid_ready['hand'].fillna(50.0).values
                grid_ready['hand_cold_pool'] = np.exp(-hand_g / 30.0) * decoupling_g
            if 'tpi_2000' in grid_ready.columns:
                grid_ready['wind_exposure'] = grid_ready['tpi_2000'].fillna(0).values * wind_g
            if 'canopy_height' in grid_ready.columns:
                canopy_g = grid_ready['canopy_height'].fillna(0).values
                grid_ready['canopy_trapping'] = np.log1p(canopy_g) * decoupling_g

            print(f"[Pro] Grid NWP local error: mean={grid_nwp_error.mean():.2f}°C")
            print(f"[Pro] Grid decoupling: mean={decoupling_g.mean():.3f}")
            
            # Enhanced grid features
            if True:
                # nwp_t2m_anomaly for grid
                _grid_trend = grid_ready[TREND_FEATURES].fillna(0).values
                _grid_trend_pred = _quick_huber.predict(_grid_trend)
                grid_ready['nwp_t2m_anomaly'] = nwp_flat - _grid_trend_pred
                print(f"[Pro v2] Grid nwp_t2m_anomaly: mean={grid_ready['nwp_t2m_anomaly'].mean():+.2f}°C")
                
                # nwp_regime_stability for grid (IDW from station values)
                grid_stab = (w * nwp_regime_stab[_trusted_idx[grid_idxs]]).sum(axis=1) / w_sum.ravel()
                grid_ready['nwp_regime_stability'] = grid_stab
                
                # ICON-EU grid features
                icon_grid = None
                try:
                    from src.fetch_icon import get_icon_grid
                    icon_grid = get_icon_grid(gx, gy, target_time=_nwp_target_time)
                    if icon_grid is not None:
                        if 't2m' in icon_grid:
                            icon_flat = icon_grid['t2m'][mask]
                            grid_ready['icon_t2m'] = icon_flat
                            # Model agreement
                            grid_ready['nwp_model_agreement'] = np.abs(nwp_flat - icon_flat)
                            print(f"[Pro v2] Grid model agreement: mean={grid_ready['nwp_model_agreement'].mean():.2f}°C")
                        if 'cloud' in icon_grid:
                            grid_ready['icon_cloud'] = icon_grid['cloud'][mask]
                        if 'wind' in icon_grid:
                            grid_ready['icon_wind'] = icon_grid['wind'][mask]
                        if 'hsurf' in icon_grid and 'dem' in grid_ready.columns:
                            icon_hsurf_flat = icon_grid['hsurf'][mask]
                            grid_ready['nwp_elev_mismatch'] = np.abs(
                                grid_ready['dem'].values - icon_hsurf_flat
                            )
                            print(f"[Pro v2] Grid elev mismatch: mean={grid_ready['nwp_elev_mismatch'].mean():.0f}m")
                except Exception as e:
                    print(f"[Pro v2] ⚠ ICON-EU grid failed: {e}")

                # Grid inversion_strength (T850 - T2m)
                _grid_inv_src = None
                t850_grid = get_nwp_t850_grid(gx, gy, target_time=_nwp_target_time)
                if t850_grid is not None:
                    t850_flat = t850_grid[mask]
                    grid_ready['inversion_strength'] = t850_flat - nwp_flat
                    _grid_inv_src = 'HARMONIE'
                elif 'icon_t2m' in grid_ready.columns and icon_grid is not None and 't850' in icon_grid:
                    icon_t850_flat = icon_grid['t850'][mask]
                    grid_ready['inversion_strength'] = icon_t850_flat - grid_ready['icon_t2m'].values
                    _grid_inv_src = 'ICON-EU'

                if _grid_inv_src is not None:
                    inv_g = grid_ready['inversion_strength']
                    print(f"[Pro v2] Grid inversion_strength ({_grid_inv_src}): "
                          f"mean={inv_g.mean():+.1f}°C")
                else:
                    grid_ready['inversion_strength'] = np.nan
        else:
            print("[Pro] ⚠️ NWP grid unavailable, predictions use Standard features only")
    
    # Grid-level calm_clear_night
    if 'calm_clear_night' in env_cols and 'nwp_cloud' in grid_ready.columns:
        grid_solar = compute_solar_elevation(grid_ready, data_fetch_time)
        g_cloud = grid_ready['nwp_cloud'].fillna(0.5)
        g_wind = grid_ready.get('nwp_wind', pd.Series(3.0, index=grid_ready.index))
        grid_ready['calm_clear_night'] = (
            (g_cloud < 0.3) &
            (g_wind.fillna(3.0) < 3.0) &
            (grid_solar < -6.0)
        ).astype(float)
        grid_night = int(grid_ready['calm_clear_night'].sum())
        print(f"[Pro] Grid calm_clear_night: {grid_night}/{len(grid_ready)} points")
    
    # Grid-level temporal and seasonal features
    if 't_prev' in env_cols:
        if prev_raster is not None:
            grid_prev = _sample_raster_at_points(prev_raster, grid_ready)
            valid_grid_prev = np.isfinite(grid_prev).sum()
            grid_ready['t_prev'] = grid_prev
            print(f"[Temporal] Grid: sampled previous output at {valid_grid_prev}/{len(grid_ready)} points")
        else:
            grid_ready['t_prev'] = np.nan
    
    # Day-of-year: same scalar values as stations
    if 'day_of_year_sin' in env_cols:
        doy = datetime.now().timetuple().tm_yday
        grid_ready['day_of_year_sin'] = np.sin(2 * np.pi * doy / 365.25)
        grid_ready['day_of_year_cos'] = np.cos(2 * np.pi * doy / 365.25)
    
    # Grid prediction, model outputs absolute temperature directly
    print(f"Predicting on {len(grid_ready):,} points...")
    g_pred, g_unc = model.predict_with_uncertainty(grid_ready)
    
    # Map to 2D
    temp_grid = np.full(mask.shape, np.nan)
    unc_grid = np.full(mask.shape, np.nan)
    temp_grid[mask] = g_pred
    unc_grid[mask] = g_unc

    # 2D DEM grid for visualization (hillshade overlay)
    dem_2d = None
    if 'dem' in grid_ready.columns:
        dem_2d = np.full(mask.shape, np.nan)
        dem_2d[mask] = grid_ready['dem'].values

    if test_metrics is not None:
        perf_label = f"Test RMSE: {test_metrics['RMSE']:.2f}°C"
    else:
        perf_label = f"{len(stations_gdf)} stations"
    title_suffix_str = f" | {perf_label}"

    # Post-processing
    if APPLY_SMOOTHING:
        print(f"Applying Gaussian smoothing (sigma={SMOOTHING_SIGMA})...")
        temp_grid = nan_gaussian_filter(temp_grid, SMOOTHING_SIGMA)
        unc_grid = nan_gaussian_filter(unc_grid, SMOOTHING_SIGMA)
        title_suffix_str += f" | Smooth $\\sigma$={SMOOTHING_SIGMA}"
    
    print(f"Grid Range: {np.nanmin(temp_grid):.1f} to {np.nanmax(temp_grid):.1f}°C")
    
    # Sanity check against training data
    train_range = (stations_gdf['temp'].min(), stations_gdf['temp'].max())
    if np.nanmin(temp_grid) < train_range[0] - 15 or np.nanmax(temp_grid) > train_range[1] + 15:
        print("⚠️ WARNING: Grid predictions show extreme extrapolation. Check inputs.")

    # Export
    export_temperature_products(temp_grid, unc_grid, gx, gy, test_metrics)
    
    # Visualization
    region_display = get_region_display_name(INTERPOLATION_REGION)
        
    if INTERPOLATION_REGION.lower() != "poland":
        region_geom = get_active_geometry(buffered=False)
        vis_stations = stations_gdf.to_crs(CRS_POLAND)
        vis_stations = vis_stations[vis_stations.geometry.within(region_geom)].copy()
        vis_stations = vis_stations.to_crs(CRS_WGS84)
        print(f"Visualization: {len(vis_stations)} stations are within {INTERPOLATION_REGION}")
    else:
        vis_stations = stations_gdf
    
    plot_temperature_map(glon, glat, temp_grid, vis_stations, OUTPUT_PLOT, show=False, title_suffix=f" | {perf_label}", region_name=region_display, data_fetch_time=data_fetch_time)
    plot_uncertainty_map(glon, glat, unc_grid, output_path=OUTPUT_UNCERTAINTY, title="Ensemble Prediction Uncertainty")
    
    # Visualization data preparation for comparison plots
    if run_benchmark and comp_df is not None:
        test_res_base = test_gdf.copy(); test_res_base['predicted'] = base_test_pred
        test_res_hyb = test_gdf.copy(); test_res_hyb['predicted'] = test_pred
        try:
            plot_model_comparison(comp_df, test_res_base, test_res_hyb)
            create_comparison_summary_image(comp_df)
        except Exception as e: print(f"Plotting error: {e}")
    try:
        plot_feature_importance(imp_df)
    except Exception as e: print(f"Feature importance plot error: {e}")

    print(f"\n✅ Pipeline Complete. Results in {RUN_OUTPUT_DIR}")
    
    # cleanup memory
    del grid_raw, grid_ready, temp_grid, unc_grid
    gc.collect()

if __name__ == "__main__":
    main()