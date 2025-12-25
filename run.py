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

from src.config import (
    GRID_RESOLUTION, CRS_POLAND, CRS_WGS84,
    TEST_SIZE, VAL_SIZE, RANDOM_STATE,
    OUTPUT_DIR, RUN_OUTPUT_DIR, SPATIAL_BUFFER_KM,
    USE_ENSEMBLE, ENSEMBLE_N_MODELS, ENSEMBLE_SEEDS,
    TREND_FEATURES, LIGHTGBM_PARAMS, OUTPUT_PLOT, OUTPUT_UNCERTAINTY,
    APPLY_SMOOTHING, SMOOTHING_SIGMA
)
from src.fetch_data import fetch_all_data
from src.prepare_features import prepare_station_data, perform_spatial_qc
from src.feature_engineering import engineer_all_features
from src.models import SimpleKrigingBaseline, EnsembleHybridModel
from src.evaluate import evaluate_predictions, print_metrics, compare_models, print_model_comparison
from src.visualize import plot_temperature_map, plot_feature_importance, plot_uncertainty_map, plot_model_comparison, create_comparison_summary_image
from src.export_utils import export_temperature_products
from src.utils import PL_GEOMETRY_2180, nan_gaussian_filter

warnings.filterwarnings("ignore")

# Grid and splut utilities
def create_prediction_grid(resolution=GRID_RESOLUTION):
    """Create empty grid within Poland."""
    print(f"\nCreating {resolution}m Grid...")
    bounds = PL_GEOMETRY_2180.bounds
    grid_x_1d = np.arange(bounds[0], bounds[2], resolution)
    grid_y_1d = np.arange(bounds[3], bounds[1], -resolution) # top to bottom
    grid_x, grid_y = np.meshgrid(grid_x_1d, grid_y_1d)
    
    transform = from_bounds(grid_x_1d[0], grid_y_1d[-1], grid_x_1d[-1], grid_y_1d[0], len(grid_x_1d), len(grid_y_1d))
    
    # WGS84 for plotting
    transformer = Transformer.from_crs(CRS_POLAND, CRS_WGS84, always_xy=True)
    grid_lon, grid_lat = transformer.transform(grid_x, grid_y)
    
    # Poland mask
    poland_mask = rasterize([(PL_GEOMETRY_2180, 1)], out_shape=grid_x.shape, transform=transform, fill=0, dtype='uint8').astype(bool)
    
    # Valid points GDF
    valid_x, valid_y = grid_x[poland_mask], grid_y[poland_mask]
    grid_gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(valid_x, valid_y)], crs=CRS_POLAND)
    
    print(f"Grid: {grid_x.shape}, Active Points: {len(grid_gdf):,}")
    return grid_gdf, grid_x_1d, grid_y_1d, grid_lon, grid_lat, poland_mask

def extract_grid_features_safe(grid_gdf, train_gdf, all_needed_cols):
    """Extract features for grid, filling missing with training medians."""
    print("\nExtracting Grid Features...")
    from src.feature_engineering import RasterFeatureExtractor, extract_terrain_features, add_coordinate_features, compute_distance_features, create_feature_interactions
    
    extractor = RasterFeatureExtractor()
    
    # Base and Terrain
    basic = extractor.extract_all_basic_features(grid_gdf, debug=False)
    for c in basic.columns: grid_gdf[c] = basic[c]
    
    # Ensure DEM exists for derivatives
    if 'dem' in grid_gdf.columns:
        grid_gdf['dem'] = grid_gdf['dem'].fillna(train_gdf['dem'].median())

    terrain = extract_terrain_features(grid_gdf, extractor)
    coords = add_coordinate_features(grid_gdf) # gets x_pl, y_pl
    dists = compute_distance_features(grid_gdf, extractor)
    
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
    
    # Data & Features
    raw_data = fetch_all_data()
    if len(raw_data) < 50: return
    
    stations_gdf = prepare_station_data(raw_data)
    
    # Engineer features
    stations_gdf, all_eng_cols = engineer_all_features(stations_gdf)
    
    # Spatial QC
    # Remove sensors that disagree with their neighbors
    stations_gdf = perform_spatial_qc(stations_gdf)
    
    # Feature organization
    # DEM + coords
    trend_cols = TREND_FEATURES 
    # Everything else engineered
    excluded = set(['station', 'temp', 'geometry', 'source', 'lat', 'lon'] + trend_cols)
    env_cols = [c for c in all_eng_cols if c not in excluded]
    
    print(f"\nFeatures: Trend={trend_cols}, EnvML={len(env_cols)} features")
    
    # Splitting + Training
    train_gdf, val_gdf, test_gdf = spatial_train_val_test_split(stations_gdf, TEST_SIZE, VAL_SIZE, SPATIAL_BUFFER_KM)
    
    # Baseline
    print("\n--- Training Baseline (Kriging) ---")
    baseline = SimpleKrigingBaseline().fit(train_gdf)
    base_test_pred = baseline.predict(test_gdf)
    base_metrics = evaluate_predictions(test_gdf['temp'], base_test_pred)
    print_metrics(base_metrics, "Baseline Test Results")
    
    # Ensemble
    print("\n--- Training Ensemble (Robust Stacking) ---")
    model = EnsembleHybridModel(trend_cols, env_cols, ENSEMBLE_N_MODELS, ENSEMBLE_SEEDS, lgbm_params=LIGHTGBM_PARAMS)
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
    
    # Importance
    imp_df = model.get_feature_importance()
    print(f"\nTop Features:\n{imp_df.head(10).to_string(index=False)}")

    # Grid Prediction and export
    print("\n--- Grid Prediction ---")
    grid_raw, gx, gy, glon, glat, mask = create_prediction_grid(GRID_RESOLUTION)
    grid_ready = extract_grid_features_safe(grid_raw, train_gdf, trend_cols + env_cols)
    
    print(f"Predicting on {len(grid_ready):,} points...")
    g_pred, g_unc = model.predict_with_uncertainty(grid_ready)
    
    # Map to 2D
    temp_grid = np.full(mask.shape, np.nan)
    unc_grid = np.full(mask.shape, np.nan)
    temp_grid[mask] = g_pred
    unc_grid[mask] = g_unc

    perf_label = f"Test RMSE: {test_metrics['RMSE']:.2f}°C (Baseline: {base_metrics['RMSE']:.2f}°C)"
    title_suffix_str = f" | {perf_label}"
    
    # Post-processing
    if APPLY_SMOOTHING:
        print(f"Applying intelligent Gaussian smoothing (sigma={SMOOTHING_SIGMA})...")
        temp_grid = nan_gaussian_filter(temp_grid, SMOOTHING_SIGMA)
        unc_grid = nan_gaussian_filter(unc_grid, SMOOTHING_SIGMA)
        title_suffix_str += f" | Smooth $\sigma$={SMOOTHING_SIGMA}"
    
    print(f"Grid Range: {np.nanmin(temp_grid):.1f} to {np.nanmax(temp_grid):.1f}°C")
    
    # Sanity check against training data
    train_range = (train_gdf['temp'].min(), train_gdf['temp'].max())
    if np.nanmin(temp_grid) < train_range[0] - 15 or np.nanmax(temp_grid) > train_range[1] + 15:
        print("⚠️ WARNING: Grid predictions show extreme extrapolation. Check inputs.")

    # Export
    export_temperature_products(temp_grid, unc_grid, gx, gy, test_metrics)
    
    # Visualization
    plot_temperature_map(glon, glat, temp_grid, stations_gdf, OUTPUT_PLOT, show=False, title_suffix=f" | {perf_label}")
    plot_uncertainty_map(glon, glat, unc_grid, output_path=OUTPUT_UNCERTAINTY, title="Ensemble Prediction Uncertainty")
    
    # Visualization data preparation for comparison plots
    test_res_base = test_gdf.copy(); test_res_base['predicted'] = base_test_pred
    test_res_hyb = test_gdf.copy(); test_res_hyb['predicted'] = test_pred
    try:
        plot_model_comparison(comp_df, test_res_base, test_res_hyb)
        create_comparison_summary_image(comp_df)
        plot_feature_importance(imp_df)
    except Exception as e: print(f"Plotting error: {e}")

    print(f"\n✅ Pipeline Complete. Results in {RUN_OUTPUT_DIR}")
    
    # cleanup memory
    del grid_raw, grid_ready, temp_grid, unc_grid
    gc.collect()

if __name__ == "__main__":
    main()