"""
Prepare station data. Geocoding and Spatial Quality Control (QC).
This QC step is really important for eliminating artifacts from bad sensors.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import HuberRegressor
from tqdm import tqdm

from .config import (
    CRS_WGS84, CRS_POLAND,
    PERFORM_SPATIAL_QC, QC_NEIGHBORS, QC_ABS_THRESHOLD,
    USE_DYNAMIC_LAPSE_RATE, MIN_STATIONS_FOR_DYNAMIC_LR, MIN_ELEVATION_SPREAD, STANDARD_LAPSE_RATE
)
from .utils import geocode_station, clean_station_name

def geocode_stations(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """Geocode stations that don't have lat/lon yet."""
    df = df.copy()
    needs_geocoding = df['lat'].isna() | df['lon'].isna()
    to_geocode = df[needs_geocoding]
    
    if len(to_geocode) == 0:
        return df
    
    print(f"\nGeocoding {len(to_geocode)} stations...")
    success = 0
    
    for idx in tqdm(to_geocode.index, desc="Geocoding"):
        station_name = clean_station_name(df.loc[idx, 'station'])
        source = df.loc[idx, 'source']
        # Hint province for IMGW to improve accuracy
        prov_hint = None # could extract from IMGW metadata if available
            
        coords, status = geocode_station(station_name, province=prov_hint, debug=debug)
        
        if status == "OK" and coords is not None:
            df.loc[idx, 'lat'] = coords[0]
            df.loc[idx, 'lon'] = coords[1]
            success += 1
            
    print(f"✓ Geocoding complete: {success} resolved")
    return df.dropna(subset=['lat', 'lon'])

def calculate_dynamic_lapse_rate(gdf: gpd.GeoDataFrame) -> float:
    """
    Calculate dynamic lapse rate from credible IMGW observations.
    """
    # Filter to credible IMGW observations only
    if 'isModel' not in gdf.columns:
        print("   ⚠️ isModel column wasn't found. Cannot calculate dynamic lapse rate.")
        return STANDARD_LAPSE_RATE
    
    credible = gdf[(gdf['source'] == 'IMGW') & (gdf['isModel'] == False)].copy()
    
    if len(credible) < MIN_STATIONS_FOR_DYNAMIC_LR:
        print(f"   ⚠️ Only {len(credible)} credible stations found (need {MIN_STATIONS_FOR_DYNAMIC_LR}). Using fallback.")
        return STANDARD_LAPSE_RATE
    
    # check elevation spread
    if 'dem' not in credible.columns:
        print("   ⚠️ No DEM data available. Using fallback.")
        return STANDARD_LAPSE_RATE
    
    elev_min = credible['dem'].min()
    elev_max = credible['dem'].max()
    elev_spread = elev_max - elev_min
    
    if elev_spread < MIN_ELEVATION_SPREAD:
        print(f"   ⚠️ Elevation spread {elev_spread:.0f}m < {MIN_ELEVATION_SPREAD}m . Using fallback.")
        return STANDARD_LAPSE_RATE
    
    # prepare the data
    X = credible['dem'].values.reshape(-1, 1)
    y = credible['temp'].values
    
    # Huber
    try:
        model = HuberRegressor(epsilon=1.35, max_iter=200)
        model.fit(X, y)
        
        slope = model.coef_[0]
        lapse_rate = -slope
        
        # Sanity check
        if lapse_rate < -0.015 or lapse_rate > 0.015:
            print(f"   ⚠️ Calculated lapse rate {lapse_rate:.4f} is out of range. Using fallback.")
            return STANDARD_LAPSE_RATE
        
        print(f"   ✅ Dynamic lapse rate: {lapse_rate:.4f} °C/m (from {len(credible)} stations, {elev_spread:.0f}m spread)")
        
        return lapse_rate
        
    except Exception as e:
        print(f"   ⚠️ Regression failed: {e}. Using fallback.")
        return STANDARD_LAPSE_RATE


def perform_spatial_qc(gdf: gpd.GeoDataFrame, lapse_rate: float = None) -> gpd.GeoDataFrame:
    """
    Advanced Spatial Outlier Detection.
    Compares each station to its neighbors, adjusting for elevation.
    Removes stations that are statistically inconsistent with their surroundings.
    """
    if not PERFORM_SPATIAL_QC or len(gdf) < QC_NEIGHBORS * 2 or 'dem' not in gdf.columns:
        print("\n⚠️  Skipping Spatial QC (not enabled, too few points, or no DEM).")
        return gdf

    print(f"\nPerforming Spatial Quality Control...")
    
    # Determine which lapse rate to use
    if lapse_rate is not None:
        effective_lapse_rate = lapse_rate
        print(f"   Using provided lapse rate: {effective_lapse_rate:.4f} °C/m")
    elif USE_DYNAMIC_LAPSE_RATE:
        print("   Calculating dynamic lapse rate based on credible IMGW observations...")
        effective_lapse_rate = calculate_dynamic_lapse_rate(gdf)
    else:
        effective_lapse_rate = STANDARD_LAPSE_RATE
        print(f"   Using static lapse rate: {effective_lapse_rate:.4f} °C/m")
    
    print(f"   Criteria: Deviation > {QC_ABS_THRESHOLD}°C from {QC_NEIGHBORS} neighbors (elevation adjusted)")

    # Work in projected coordinates
    gdf_proj = gdf.to_crs(CRS_POLAND).copy()
    coords = np.array([[g.x, g.y] for g in gdf_proj.geometry])
    temps = gdf_proj['temp'].values
    dems = gdf_proj['dem'].values

    # Find neighbors
    nbrs = NearestNeighbors(n_neighbors=QC_NEIGHBORS + 1, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    outlier_mask = np.zeros(len(gdf), dtype=bool)
    deviations = []
    
    # Track credible stations
    trusted_count = 0
    has_isModel = 'isModel' in gdf.columns

    for i in range(len(gdf)):
        row = gdf.iloc[i]
        
        # Bypass QC for credible IMGW observations
        if has_isModel and row['source'] == 'IMGW' and row['isModel'] == False:
            deviations.append(0.0)
            trusted_count += 1
            continue
        
        # Neighbors (excluding self, which is index 0)
        nbr_indices = indices[i, 1:]
        
        my_temp = temps[i]
        my_dem = dems[i]
        
        nbr_temps = temps[nbr_indices]
        nbr_dems = dems[nbr_indices]
        
        # Adjust neighbor temperatures to the elevation of the current station
        elev_diffs = my_dem - nbr_temps
        # Use effective lapse rate for elevation adjustment
        adjusted_nbr_temps = nbr_temps - (nbr_dems - my_dem) * effective_lapse_rate
        
        # Expected temperature is median of adjusted neighbors (robust to neighbor outliers)
        expected_temp = np.nanmedian(adjusted_nbr_temps)
        
        deviation = np.abs(my_temp - expected_temp)
        deviations.append(deviation)
        
        # Flag if deviation is too high
        if deviation > QC_ABS_THRESHOLD:
            outlier_mask[i] = True

    num_outliers = np.sum(outlier_mask)
    
    if trusted_count > 0:
        print(f"   ✅ {trusted_count} credible IMGW observations successfully bypassed QC")
    
    if num_outliers > 0:
        # print some examples
        outlier_indices = np.where(outlier_mask)[0]
        print(f"   ❌ Detected {num_outliers} spatial outliers (measurement errors):")
        for idx in outlier_indices[:5]: # show first 5
            row = gdf.iloc[idx]
            print(f"      - {row['station']} ({row['source']}): "
                  f"T={row['temp']:.1f}°C, Dev={deviations[idx]:.1f}°C")
        if num_outliers > 5: print(f"      ... and {num_outliers-5} more.")
            
        # Remove outliers
        gdf_clean = gdf[~outlier_mask].copy()
        print(f"✓ QC Complete. Kept {len(gdf_clean)}/{len(gdf)} stations.")
        return gdf_clean
    else:
        print("✓ QC Complete. No outliers detected.")
        return gdf

def prepare_station_data(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert DataFrame to GeoDataFrame, geocode, and remove duplicates."""
    # Basic clean
    df = df.dropna(subset=['temp'])
    
    # Geocode
    df = geocode_stations(df)
    
    # Create GeoDataFrame
    geometry = [Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=CRS_WGS84)
    
    # Deduplicate locations (prioritize IMGW over Netatmo PWS if at same spot)
    # sort by source priority helps keep better stations
    source_priority = {'IMGW': 1, 'TRAX': 2, 'NETATMO': 3}
    gdf['priority'] = gdf['source'].map(source_priority).fillna(99)
    gdf = gdf.sort_values('priority')
    
    # Keep first (highest priority) at unique lat/lon
    before = len(gdf)
    gdf = gdf.drop_duplicates(subset=['lat', 'lon'], keep='first')
    gdf = gdf.drop(columns=['priority'])
    
    print(f"\n✓ Prepared {len(gdf)} unique stations (dropped {before-len(gdf)} duplicates)")
        
    return gdf