"""
Feature engineering with detailed debugging for raster extraction.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from scipy import ndimage
from typing import Dict, List, Tuple
import warnings

from .config import (
    RASTER_FILES, CRS_POLAND, CRS_WGS84,
    EXTRACT_TERRAIN_DERIVATIVES, TERRAIN_WINDOW_SIZES,
    USE_SPATIAL_LAG_FEATURES, SPATIAL_LAG_NEIGHBORS
)

warnings.filterwarnings('ignore')

class RasterFeatureExtractor:
    """Extract features from raster files at point locations"""
    
    def __init__(self):
        self.rasters = {}
        self.transforms = {}
        self.crs_list = {}
        self.nodata_values = {}
        self.bounds = {}
        self.load_rasters()
    
    def load_rasters(self):
        """Load all available rasters"""
        print("\nLoading raster data...")
        
        for name, path in RASTER_FILES.items():
            if not path.exists():
                print(f"   ⚠️  Skipping {name}: file not found")
                continue
            
            try:
                with rasterio.open(path) as src:
                    self.rasters[name] = src.read(1)
                    self.transforms[name] = src.transform
                    self.crs_list[name] = src.crs
                    self.nodata_values[name] = src.nodata
                    self.bounds[name] = src.bounds
                    
                    print(f"   ✓ {name}: {self.rasters[name].shape}, "
                          f"CRS: {src.crs}, "
                          f"Bounds: ({src.bounds.left:.2f}, {src.bounds.bottom:.2f}) to "
                          f"({src.bounds.right:.2f}, {src.bounds.top:.2f})")
            except Exception as e:
                print(f"   ❌ Failed to load {name}: {e}")
    
    def extract_at_points(self, gdf: gpd.GeoDataFrame, raster_name: str, debug: bool = False) -> np.ndarray:
        """Extract raster values at point locations with detailed debugging"""
        if raster_name not in self.rasters:
            return np.full(len(gdf), np.nan)
        
        raster = self.rasters[raster_name]
        transform = self.transforms[raster_name]
        raster_crs = self.crs_list[raster_name]
        nodata = self.nodata_values[raster_name]
        bounds = self.bounds[raster_name]
        
        # Ensure GeoDataFrame is in the same CRS as raster
        if gdf.crs != raster_crs:
            gdf_reproj = gdf.to_crs(raster_crs)
        else:
            gdf_reproj = gdf
        
        values = []
        in_bounds_count = 0
        valid_value_count = 0
        
        # sample first few points (debug)
        if debug and len(gdf_reproj) > 0:
            first_geom = gdf_reproj.geometry.iloc[0]
            print(f"\n   DEBUG {raster_name}:")
            print(f"     First point: ({first_geom.x:.4f}, {first_geom.y:.4f})")
            print(f"     Raster bounds: ({bounds.left:.4f}, {bounds.bottom:.4f}) to ({bounds.right:.4f}, {bounds.top:.4f})")
            print(f"     Raster shape: {raster.shape}")
        
        for idx, geom in enumerate(gdf_reproj.geometry):
            # check if point is within raster bounds
            if not (bounds.left <= geom.x <= bounds.right and 
                    bounds.bottom <= geom.y <= bounds.top):
                values.append(np.nan)
                continue
            
            in_bounds_count += 1
            
            try:
                # get row/col
                row_float, col_float = rowcol(transform, geom.x, geom.y)
                row = int(row_float)
                col = int(col_float)
                
                # debug first few
                if debug and idx < 3:
                    print(f"     Point {idx}: ({geom.x:.4f}, {geom.y:.4f}) -> row={row}, col={col}")
                
                # check array bounds
                if 0 <= row < raster.shape[0] and 0 <= col < raster.shape[1]:
                    val = float(raster[row, col])
                    
                    # check for nodata
                    is_valid = True
                    if nodata is not None and val == nodata:
                        is_valid = False
                    elif np.isnan(val):
                        is_valid = False
                    elif val < -9000 or val > 1e6:
                        is_valid = False
                    
                    if is_valid:
                        valid_value_count += 1
                        values.append(val)
                        if debug and idx < 3:
                            print(f"       -> value = {val:.2f} ✓")
                    else:
                        values.append(np.nan)
                        if debug and idx < 3:
                            print(f"       -> nodata (val={val})")
                else:
                    values.append(np.nan)
                    if debug and idx < 3:
                        print(f"       -> out of raster bounds")
            except Exception as e:
                values.append(np.nan)
                if debug and idx < 3:
                    print(f"       -> ERROR: {e}")
        
        if debug:
            print(f"     In bounds: {in_bounds_count}/{len(gdf)}")
            print(f"     Valid values: {valid_value_count}/{len(gdf)}")
        
        return np.array(values)
    
    def extract_all_basic_features(self, gdf: gpd.GeoDataFrame, debug: bool = True) -> pd.DataFrame:
        """Extract all basic raster features"""
        print("\nExtracting basic raster features...")
        
        features = pd.DataFrame(index=gdf.index)
        
        for i, raster_name in enumerate(self.rasters.keys()):
            # debug first raster in detail
            values = self.extract_at_points(gdf, raster_name, debug=(i == 0 and debug))
            features[raster_name] = values
            valid_pct = 100 * (~np.isnan(values)).sum() / len(values)
            print(f"   {raster_name:15s}: {valid_pct:5.1f}% valid ({(~np.isnan(values)).sum()}/{len(values)})")
        
        return features

class TerrainAnalyzer:
    """Compute terrain derivatives from DEM"""
    
    def __init__(self, dem: np.ndarray, transform):
        self.dem = dem
        self.transform = transform
        self.pixel_size = abs(transform[0])
        
        # Replace NaN with local median for gradient computation
        self.dem_filled = dem.copy()
        if np.isnan(self.dem_filled).any():
            from scipy.ndimage import median_filter
            # fill NaN with median of neighbors
            mask = np.isnan(self.dem_filled)
            self.dem_filled[mask] = np.nanmedian(self.dem_filled)
    
    def compute_slope(self) -> np.ndarray:
        """Compute slope in degrees"""
        dy, dx = np.gradient(self.dem_filled, self.pixel_size)
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        # mask where original DEM was NaN
        slope[np.isnan(self.dem)] = np.nan
        return slope
    
    def compute_aspect(self) -> np.ndarray:
        """Compute aspect in degrees (0-360)"""
        dy, dx = np.gradient(self.dem_filled, self.pixel_size)
        aspect = np.degrees(np.arctan2(-dx, dy))
        aspect = (aspect + 360) % 360
        aspect[np.isnan(self.dem)] = np.nan
        return aspect
    
    def compute_curvature(self) -> np.ndarray:
        """Compute profile curvature"""
        dy, dx = np.gradient(self.dem_filled, self.pixel_size)
        dyy, dyx = np.gradient(dy, self.pixel_size)
        dxy, dxx = np.gradient(dx, self.pixel_size)
        curvature = dxx + dyy
        curvature[np.isnan(self.dem)] = np.nan
        return curvature
    
    def compute_tpi(self, window_size: int = 9) -> np.ndarray:
        """
        Topographic Position Index calculation
        """
        # use nanmean-like behavior
        from scipy.ndimage import generic_filter
        
        def nan_mean(values):
            """Mean ignoring NaN"""
            valid = values[~np.isnan(values)]
            return valid.mean() if len(valid) > 0 else np.nan
        
        # compute local mean elevation
        mean_elevation = generic_filter(
            self.dem,
            nan_mean,
            size=window_size,
            mode='constant',
            cval=np.nan
        )
        
        tpi = self.dem - mean_elevation
        return tpi
    
    def compute_roughness(self, window_size: int = 9) -> np.ndarray:
        """
        Terrain roughness (std dev of elevation) calculation
        """
        from scipy.ndimage import generic_filter
        
        def nan_std(values):
            """Std dev ignoring NaN"""
            valid = values[~np.isnan(values)]
            return valid.std() if len(valid) > 1 else 0.0
        
        roughness = generic_filter(
            self.dem,
            nan_std,
            size=window_size,
            mode='constant',
            cval=np.nan
        )
        
        return roughness

def extract_terrain_features(gdf: gpd.GeoDataFrame, extractor: RasterFeatureExtractor) -> pd.DataFrame:
    """Extract advanced terrain features"""
    if 'dem' not in extractor.rasters or not EXTRACT_TERRAIN_DERIVATIVES:
        print("   ⚠️  Skipping terrain derivatives")
        return pd.DataFrame(index=gdf.index)
    
    print("\nComputing terrain derivatives...")
    
    dem = extractor.rasters['dem']
    transform = extractor.transforms['dem']
    analyzer = TerrainAnalyzer(dem, transform)
    
    # Only compute if DEM has valid data
    if np.all(np.isnan(dem)) or np.all(dem == extractor.nodata_values.get('dem', -9999)):
        print("   ⚠️  DEM has no valid data, skipping terrain derivatives")
        return pd.DataFrame(index=gdf.index)
    
    terrain_layers = {
        'slope': analyzer.compute_slope(),
        'aspect': analyzer.compute_aspect(),
        'curvature': analyzer.compute_curvature(),
    }
    
    # Multi-scale TPI and roughness
    for window in TERRAIN_WINDOW_SIZES:
        terrain_layers[f'tpi_{window}'] = analyzer.compute_tpi(window)
        terrain_layers[f'roughness_{window}'] = analyzer.compute_roughness(window)
    
    print(f"   ✓ Computed {len(terrain_layers)} terrain layers")
    
    # Extract at points
    features = pd.DataFrame(index=gdf.index)
    
    for name, layer in terrain_layers.items():
        extractor.rasters[name] = layer
        extractor.transforms[name] = transform
        extractor.crs_list[name] = extractor.crs_list['dem']
        extractor.nodata_values[name] = None
        extractor.bounds[name] = extractor.bounds['dem']
        
        features[name] = extractor.extract_at_points(gdf, name, debug=False)
        
        # cleanup
        del extractor.rasters[name]
        del extractor.transforms[name]
        del extractor.crs_list[name]
        del extractor.nodata_values[name]
        del extractor.bounds[name]
    
    # cyclic aspect encoding
    if 'aspect' in features.columns:
        features['aspect_sin'] = np.sin(np.radians(features['aspect']))
        features['aspect_cos'] = np.cos(np.radians(features['aspect']))
        features = features.drop('aspect', axis=1)
    
    valid_count = features.notna().sum().sum()
    total_count = features.size
    print(f"   ✓ Extracted {len(features.columns)} terrain features ({100*valid_count/total_count:.1f}% valid)")
    
    return features

def compute_spatial_lag_features(gdf: gpd.GeoDataFrame, n_neighbors: int = SPATIAL_LAG_NEIGHBORS) -> pd.DataFrame:
    """Compute spatial lag features"""
    if not USE_SPATIAL_LAG_FEATURES or 'temp' not in gdf.columns:
        return pd.DataFrame(index=gdf.index)
    
    print(f"\nComputing spatial lag features ({n_neighbors} neighbors)...")
    
    from sklearn.neighbors import NearestNeighbors
    
    gdf_proj = gdf.to_crs(CRS_POLAND)
    coords = np.array([[geom.x, geom.y] for geom in gdf_proj.geometry])
    temps = gdf['temp'].values
    
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    knn.fit(coords)
    distances, indices = knn.kneighbors(coords)
    
    lag_features = pd.DataFrame(index=gdf.index)
    
    for i in range(len(coords)):
        neighbor_dists = distances[i, 1:]
        neighbor_temps = temps[indices[i, 1:]]
        
        weights = 1.0 / (neighbor_dists + 1.0)
        weights /= weights.sum()
        
        lag_features.loc[gdf.index[i], 'spatial_lag_temp'] = (weights * neighbor_temps).sum()
        lag_features.loc[gdf.index[i], 'spatial_lag_min_dist'] = neighbor_dists.min()
        lag_features.loc[gdf.index[i], 'spatial_lag_mean_dist'] = neighbor_dists.mean()
    
    print(f"   ✓ Computed {len(lag_features.columns)} spatial lag features")
    
    return lag_features

def add_coordinate_features(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Add coordinate-based features
    """
    print("\nAdding coordinate features...")
    
    features = pd.DataFrame(index=gdf.index)
    
    # Get WGS84 coordinates
    gdf_wgs84 = gdf.to_crs(CRS_WGS84) if gdf.crs != CRS_WGS84 else gdf
    features['lat'] = [geom.y for geom in gdf_wgs84.geometry]
    features['lon'] = [geom.x for geom in gdf_wgs84.geometry]
    
    # Get projected coordinates (!)
    gdf_poland = gdf.to_crs(CRS_POLAND) if gdf.crs != CRS_POLAND else gdf
    features['x_pl'] = [geom.x for geom in gdf_poland.geometry]
    features['y_pl'] = [geom.y for geom in gdf_poland.geometry]
    
    # removed x_normalized and y_normalized to prevent stripes
    
    print(f"   ✓ Added coordinate features (x_pl, y_pl, lat, lon)")
    
    return features

def compute_distance_features(gdf: gpd.GeoDataFrame, extractor: RasterFeatureExtractor = None) -> pd.DataFrame:
    """
    Compute distance-based features that work for both stations and grid.
    """
    from .config import COMPUTE_DISTANCE_FEATURES, DISTANCE_FEATURES
    
    if not COMPUTE_DISTANCE_FEATURES:
        return pd.DataFrame(index=gdf.index)
    
    print("\nComputing distance features...")
    
    features = pd.DataFrame(index=gdf.index)
    
    # Convert to projected CRS for distance calculations
    gdf_proj = gdf.to_crs(CRS_POLAND) if gdf.crs != CRS_POLAND else gdf
    
    # Distance to coast (Baltic Sea - approximate as northern boundary)
    if DISTANCE_FEATURES.get('coast', False):
        y_coords = np.array([geom.y for geom in gdf_proj.geometry])
        coast_y = 600000  # approximate southern coast of Baltic
        features['dist_coast'] = np.maximum(coast_y - y_coords, 0) / 1000  # km
        print(f"   ✓ Distance to coast")
    
    # Distance to mountains (high elevation areas)
    if DISTANCE_FEATURES.get('mountains', False) and extractor and 'dem' in extractor.rasters:
        dem = extractor.rasters['dem']
        dem_transform = extractor.transforms['dem']
        
        # find mountain areas (elev > 800m for Poland)
        mountain_mask = dem > 800
        
        if mountain_mask.any():
            # Get coordinates of mountain pixels
            rows, cols = np.where(mountain_mask)
            
            # Convert to geographic coordinates
            from rasterio.transform import xy
            mountain_coords = []
            for row, col in zip(rows, cols):
                x, y = xy(dem_transform, row, col)
                mountain_coords.append([x, y])
            
            mountain_coords = np.array(mountain_coords)
            
            # Convert to EPSG:2180
            from shapely.geometry import Point
            mountain_points = [Point(x, y) for x, y in mountain_coords]
            mountain_gdf = gpd.GeoDataFrame(geometry=mountain_points, crs=extractor.crs_list['dem'])
            mountain_gdf = mountain_gdf.to_crs(CRS_POLAND)
            mountain_coords_proj = np.array([[g.x, g.y] for g in mountain_gdf.geometry])
            
            # Compute minimum distance to any mountain pixel
            from scipy.spatial import cKDTree
            tree = cKDTree(mountain_coords_proj)
            
            station_coords = np.array([[geom.x, geom.y] for geom in gdf_proj.geometry])
            distances, _ = tree.query(station_coords)
            
            features['dist_mountains'] = distances / 1000  # Convert to km
            print(f"   ✓ Distance to mountains ({len(mountain_coords)} mountain pixels)")
        else:
            features['dist_mountains'] = 0
            print(f"   ⚠️  No mountains found in DEM")
    
    # Elevation-derived features (smooth proxies for terrain complexity)
    if extractor and 'dem' in gdf.columns:
        # Elevation percentile (relative position in elevation range)
        elevation = gdf['dem'].values
        elev_min, elev_max = np.nanmin(elevation), np.nanmax(elevation)
        if elev_max > elev_min:
            features['elevation_percentile'] = (elevation - elev_min) / (elev_max - elev_min)
        
        # Squared elevation for non-linear lapse rate effects
        features['elevation_squared'] = (elevation / 1000) ** 2  # normalized
        
        print(f"   ✓ Elevation-derived features")
    
    print(f"   ✓ Computed {len(features.columns)} distance features")
    
    return features

def create_feature_interactions(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Create interaction features between key predictors.
    """
    from .config import CREATE_FEATURE_INTERACTIONS, INTERACTION_PAIRS
    
    if not CREATE_FEATURE_INTERACTIONS:
        return pd.DataFrame(index=gdf.index)
    
    print("\nCreating feature interactions...")
    
    features = pd.DataFrame(index=gdf.index)
    
    for feat1, feat2 in INTERACTION_PAIRS:
        if feat1 in gdf.columns and feat2 in gdf.columns:
            # Multiplicative interaction
            interaction_name = f"{feat1}_x_{feat2}"
            features[interaction_name] = gdf[feat1] * gdf[feat2]
            
            # also create squared terms for important features
            if feat1 == 'dem' and f"dem_squared" not in features.columns:
                features['dem_squared'] = gdf['dem'] ** 2
    
    print(f"   ✓ Created {len(features.columns)} interaction features")
    
    return features

def engineer_all_features(gdf: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, List[str]]:
    """Complete feature engineering pipeline"""
    print("Feature engineering pipeline")
    
    gdf = gdf.copy()
    original_cols = set(gdf.columns)
    
    # Initialize extractor
    extractor = RasterFeatureExtractor()
    
    # Extract basic raster features
    basic_features = extractor.extract_all_basic_features(gdf, debug=True)
    
    # Extract terrain derivatives
    terrain_features = extract_terrain_features(gdf, extractor)
    
    # Coordinate features
    coord_features = add_coordinate_features(gdf)
    
    # Distance features
    distance_features = compute_distance_features(gdf, extractor)
    
    # Combine base features first
    base_features = pd.concat([
        basic_features,
        terrain_features,
        coord_features,
        distance_features
    ], axis=1)
    
    for col in base_features.columns:
        gdf[col] = base_features[col]
    
    # NOW create interactions (after base features are in gdf)
    interaction_features = create_feature_interactions(gdf)
    
    for col in interaction_features.columns:
        gdf[col] = interaction_features[col]
    
    # Get all new feature columns
    feature_cols = [col for col in gdf.columns if col not in original_cols and col != 'temp']
    
    # Keep features with at least 10% valid data
    valid_features = []
    for col in feature_cols:
        valid_pct = 100 * (~gdf[col].isna()).sum() / len(gdf)
        if valid_pct >= 10:
            valid_features.append(col)
        else:
            print(f"   ⚠️  Dropping {col}: only {valid_pct:.1f}% valid")
    
    feature_cols = valid_features
    
    # Remove rows which contain too many NaNs
    nan_threshold = 0.5
    nan_counts = gdf[feature_cols].isna().sum(axis=1)
    valid_mask = nan_counts < (len(feature_cols) * nan_threshold)
    
    print(f"\nFeature engineering summary:")
    print(f"   Total features created: {len(base_features.columns) + len(interaction_features.columns)}")
    print(f"   Features kept (>10% valid): {len(feature_cols)}")
    print(f"   Stations before cleaning: {len(gdf)}")
    print(f"   Stations after cleaning: {valid_mask.sum()}")
    print(f"   Removed (too many NaNs): {(~valid_mask).sum()}")
    
    gdf = gdf[valid_mask].copy()
    
    # Fill remaining NaNs
    for col in feature_cols:
        if gdf[col].isna().any():
            median_val = gdf[col].median()
            if np.isnan(median_val):
                median_val = 0.0
            gdf[col] = gdf[col].fillna(median_val)
    
    return gdf, feature_cols