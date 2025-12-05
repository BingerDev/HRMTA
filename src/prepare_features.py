"""
Prepare station data. Geocoding and Spatial Quality Control (QC).
This QC step is really important for eliminating artifacts from bad sensors.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from .config import (
    CRS_WGS84, CRS_POLAND,
    PERFORM_SPATIAL_QC, QC_NEIGHBORS, QC_ABS_THRESHOLD, QC_LAPSE_RATE
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

def perform_spatial_qc(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Advanced Spatial Outlier Detection.
    Compares each station to its neighbors, adjusting for elevation.
    Removes stations that are statistically inconsistent with their surroundings.
    """
    if not PERFORM_SPATIAL_QC or len(gdf) < QC_NEIGHBORS * 2 or 'dem' not in gdf.columns:
        print("\n⚠️  Skipping Spatial QC (not enabled, too few points, or no DEM).")
        return gdf

    print(f"\nPerforming Spatial Quality Control...")
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

    for i in range(len(gdf)):
        # Neighbors (excluding self, which is index 0)
        nbr_indices = indices[i, 1:]
        
        my_temp = temps[i]
        my_dem = dems[i]
        
        nbr_temps = temps[nbr_indices]
        nbr_dems = dems[nbr_indices]
        
        # Adjust neighbor temperatures to the elevation of the current station
        elev_diffs = my_dem - nbr_temps
        # Standard lapse rate adjustment (approximate, but good for QC)
        adjusted_nbr_temps = nbr_temps - (nbr_dems - my_dem) * QC_LAPSE_RATE
        
        # Expected temperature is median of adjusted neighbors (robust to neighbor outliers)
        expected_temp = np.nanmedian(adjusted_nbr_temps)
        
        deviation = np.abs(my_temp - expected_temp)
        deviations.append(deviation)
        
        # Flag if deviation is too high
        if deviation > QC_ABS_THRESHOLD:
            outlier_mask[i] = True

    num_outliers = np.sum(outlier_mask)
    
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