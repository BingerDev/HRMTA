"""
Fetch ICON-EU NWP data for multi-model integration.
"""
import os
import bz2
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import requests
import geopandas as gpd

from .config import CACHE_DIR, CRS_WGS84, CRS_POLAND, ICON_CONFIG, NWP_CACHE_MAX_FILES

# Poland bounding box
POLAND_BBOX = {
    "lon_min": 13.5,
    "lon_max": 25.0,
    "lat_min": 48.5,
    "lat_max": 55.5,
}

# ICON-EU cache
_ICON_CACHE: Dict[str, Any] = {}

def _find_latest_icon_run() -> Tuple[str, int, int]:
    """
    Find the latest available ICON-EU model run.
    
    Returns:
    - Tuple of (run_date_str YYYYMMDD, run_hour, forecast_hour).
    """
    now = datetime.now(timezone.utc)
    
    # ICON-EU has ~3-4h processing delay
    available_hour = (now.hour - 4) % 24
    run_hour = (available_hour // 3) * 3
    
    # If we went back past midnight
    run_date = now
    if run_hour > now.hour:
        run_date = now - timedelta(days=1)
    
    # Calculate forecast hour to match current time
    run_dt = run_date.replace(hour=run_hour, minute=0, second=0, microsecond=0)
    target_dt = now.replace(minute=0, second=0, microsecond=0)
    forecast_hour = int((target_dt - run_dt).total_seconds() / 3600)
    forecast_hour = max(0, min(forecast_hour, 120))
    
    run_date_str = run_date.strftime("%Y%m%d")
    return run_date_str, run_hour, forecast_hour

def _download_icon_variable(variable: str, run_date: str, run_hour: int,
                            forecast_hour: int) -> Optional[Path]:
    """
    Download a single ICON-EU variable file from DWD open data.
    
    Args:
    - variable: Variable name (t_2m, clct, u_10m, v_10m, hsurf).
    - run_date: Run date as YYYYMMDD.
    - run_hour: Model run hour (0, 3, 6, ..., 21).
    - forecast_hour: Forecast lead time (0-120).
    
    Returns:
    - Path to decompressed GRIB2 file, or None if failed.
    """
    cache_dir = CACHE_DIR / "icon"
    cache_dir.mkdir(exist_ok=True, parents=True)
    
    if variable == "hsurf":
        url = (f"{ICON_CONFIG['base_url']}/{run_hour:02d}/{variable}/"
               f"icon-eu_europe_regular-lat-lon_time-invariant_"
               f"{run_date}{run_hour:02d}_HSURF.grib2.bz2")
    elif variable == "t_850":
        url = (f"{ICON_CONFIG['base_url']}/{run_hour:02d}/t/"
               f"icon-eu_europe_regular-lat-lon_pressure-level_"
               f"{run_date}{run_hour:02d}_{forecast_hour:03d}_850_T.grib2.bz2")
    else:
        var_upper = variable.upper()
        url = (f"{ICON_CONFIG['base_url']}/{run_hour:02d}/{variable}/"
               f"icon-eu_europe_regular-lat-lon_single-level_"
               f"{run_date}{run_hour:02d}_{forecast_hour:03d}_{var_upper}.grib2.bz2")
    
    cache_file = cache_dir / f"icon-eu_{variable}_{run_date}{run_hour:02d}_{forecast_hour:03d}.grib2"
    
    # Check cache
    if cache_file.exists():
        return cache_file
    
    try:
        print(f"[ICON] Downloading {variable}...")
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        
        # Decompress bz2
        compressed = resp.content
        decompressed = bz2.decompress(compressed)
        
        with open(cache_file, 'wb') as f:
            f.write(decompressed)
        
        size_mb = cache_file.stat().st_size / (1024 * 1024)
        print(f"[ICON] ✓ {variable}: {size_mb:.1f} MB")
        return cache_file
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            # Try previous run
            return None
        print(f"[ICON] ⚠ Failed to download {variable}: {e}")
        return None
    except Exception as e:
        print(f"[ICON] ⚠ Failed to download {variable}: {e}")
        return None

def _clean_icon_cache(cache_dir: Path, max_files: int = 2):
    """Remove old ICON GRIB files, keeping only the most recent per variable."""
    files = sorted(cache_dir.glob("*.grib2"), key=lambda f: f.stat().st_mtime, reverse=True)
    for old_file in files[max_files * len(ICON_CONFIG['variables']):]:
        try:
            old_file.unlink()
        except Exception:
            pass

def _parse_icon_grib(grib_path: Path, variable: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Parse an ICON-EU GRIB2 file.
        
    Returns:
    - Dict with 'data', 'lat', 'lon' arrays, or None.
    """
    try:
        import cfgrib
        
        ds = cfgrib.open_datasets(str(grib_path))
        if not ds:
            return None
        
        # Take the first dataset (single-variable files)
        dataset = ds[0]
        
        # Get the data variable (might have different names)
        data_var = list(dataset.data_vars)[0]
        data = dataset[data_var].values.squeeze()
        
        # Get coordinates
        lat = None
        lon = None
        for name in ['latitude', 'lat', 'y']:
            if name in dataset.coords:
                lat = dataset.coords[name].values
                break
        for name in ['longitude', 'lon', 'x']:
            if name in dataset.coords:
                lon = dataset.coords[name].values
                break
        
        if lat is None or lon is None:
            print(f"[ICON] ⚠ Could not find coordinates in {grib_path.name}")
            return None
        
        # Convert temperature from Kelvin to Celsius
        if variable in ('t_2m', 't_850') and np.nanmean(data) > 100:
            data = data - 273.15
        
        # Normalize cloud cover to 0-1
        if variable == 'clct' and np.nanmax(data) > 1.5:
            data = data / 100.0
        
        return {'data': data, 'lat': lat, 'lon': lon}
    except Exception as e:
        print(f"[ICON] ⚠ Failed to parse {grib_path.name}: {e}")
        return None

def _interpolate_to_points(parsed: Dict, target_lons: np.ndarray,
                           target_lats: np.ndarray) -> np.ndarray:
    """Interpolate ICON-EU regular grid to target points."""
    from scipy.interpolate import RegularGridInterpolator
    
    lat = parsed['lat']
    lon = parsed['lon']
    data = parsed['data']
    
    # Ensure lat is ascending for RegularGridInterpolator
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        data = data[::-1, :]
    
    # Create interpolator
    interp = RegularGridInterpolator(
        (lat, lon), data,
        method='cubic',
        bounds_error=False,
        fill_value=np.nan
    )

    points = np.column_stack([target_lats, target_lons])
    return interp(points)

def _interpolate_to_grid(parsed: Dict, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    """Interpolate ICON-EU regular grid to EPSG:2180 prediction grid."""
    import pyproj
    from scipy.interpolate import RegularGridInterpolator

    lat = parsed['lat']
    lon = parsed['lon']
    data = parsed['data']

    if lat[0] > lat[-1]:
        lat = lat[::-1]
        data = data[::-1, :]

    interp = RegularGridInterpolator(
        (lat, lon), data,
        method='cubic',
        bounds_error=False,
        fill_value=np.nan
    )
    
    # Transform grid from EPSG:2180 to WGS84
    transformer = pyproj.Transformer.from_crs(CRS_POLAND, CRS_WGS84, always_xy=True)
    grid_x_2d, grid_y_2d = np.meshgrid(grid_x, grid_y)
    lon_2d, lat_2d = transformer.transform(grid_x_2d, grid_y_2d)
    
    points = np.column_stack([lat_2d.ravel(), lon_2d.ravel()])
    result = interp(points).reshape(grid_x_2d.shape)
    return result

def fetch_icon_data(target_time: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    """
    Fetch ICON-EU data (temperature, cloud, wind, model orography).
    
    Returns:
    - Dict with parsed data for each variable, or None if failed.
    """
    ref_time = target_time or datetime.now(timezone.utc)
    cache_key = ref_time.strftime("%Y%m%d_%H")
    
    if cache_key in _ICON_CACHE:
        print("[ICON] Using cached ICON-EU data")
        return _ICON_CACHE[cache_key]
    
    print(f"[ICON] Fetching ICON-EU data...")
    
    # Find latest available run
    run_date, run_hour, forecast_hour = _find_latest_icon_run()
    print(f"[ICON] Run: {run_date} {run_hour:02d}Z, forecast hour: {forecast_hour}")
    
    # Try to download, with fallback to previous run
    result = {}
    variables = ['t_2m', 'clct', 'u_10m', 'v_10m', 'hsurf', 't_850']
    
    for attempt in range(3):
        missing = []
        for var in variables:
            if var in result:
                continue
            
            grib_path = _download_icon_variable(var, run_date, run_hour, forecast_hour)
            if grib_path is not None:
                parsed = _parse_icon_grib(grib_path, var)
                if parsed is not None:
                    result[var] = parsed
                    continue
            missing.append(var)
        
        if not missing or 't_2m' in result:
            break
        
        # Fall back to previous run
        prev_hour = (run_hour - 3) % 24
        if prev_hour > run_hour:
            prev_date = (datetime.strptime(run_date, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
        else:
            prev_date = run_date
        run_date, run_hour = prev_date, prev_hour
        forecast_hour = min(forecast_hour + 3, 120)
        print(f"[ICON] Retrying with run {run_date} {run_hour:02d}Z, +{forecast_hour}h")
    
    if 't_2m' not in result:
        print("[ICON] ❌ Could not fetch ICON-EU temperature")
        return None
    
    print(f"[ICON] ✓ Fetched {len(result)} variables: {list(result.keys())}")

    # Clean old GRIB files from disk cache
    if NWP_CACHE_MAX_FILES > 0:
        _clean_icon_cache(CACHE_DIR / "icon", max_files=NWP_CACHE_MAX_FILES)

    # In-memory cache
    _ICON_CACHE[cache_key] = result
    for old_key in list(_ICON_CACHE.keys()):
        if old_key != cache_key:
            del _ICON_CACHE[old_key]
    
    return result

def get_icon_at_stations(stations_gdf: gpd.GeoDataFrame,
                         target_time: Optional[datetime] = None) -> gpd.GeoDataFrame:
    """
    Add ICON-EU columns to station GeoDataFrame.
    
    Adds: icon_t2m, icon_cloud, icon_wind, icon_hsurf, icon_t850.
    
    Args:
    - stations_gdf: GeoDataFrame with station points (any CRS).
    - target_time: Locked forecast time for temporal consistency.
    
    Returns:
    - GeoDataFrame with ICON-EU columns added.
    """
    icon_data = fetch_icon_data(target_time)
    if icon_data is None:
        # Fill with NaN
        for col in ['icon_t2m', 'icon_cloud', 'icon_wind', 'icon_hsurf', 'icon_t850']:
            stations_gdf[col] = np.nan
        return stations_gdf
    
    # Ensure WGS84
    gdf_wgs = stations_gdf.to_crs(CRS_WGS84) if stations_gdf.crs and str(stations_gdf.crs) != CRS_WGS84 else stations_gdf
    lons = gdf_wgs.geometry.x.values
    lats = gdf_wgs.geometry.y.values
    
    # Interpolate each variable
    if 't_2m' in icon_data:
        vals = _interpolate_to_points(icon_data['t_2m'], lons, lats)
        stations_gdf['icon_t2m'] = vals
        valid = (~np.isnan(vals)).sum()
        print(f"[ICON] Added icon_t2m to {valid}/{len(stations_gdf)} stations")
        t2m_range = f"{np.nanmin(vals):.1f} to {np.nanmax(vals):.1f}°C"
        print(f"[ICON] ICON-EU temperature range: {t2m_range}")
    
    if 'clct' in icon_data:
        vals = np.clip(_interpolate_to_points(icon_data['clct'], lons, lats), 0.0, 1.0)
        stations_gdf['icon_cloud'] = vals
        valid = (~np.isnan(vals)).sum()
        print(f"[ICON] Added icon_cloud to {valid}/{len(stations_gdf)} stations")
    
    if 'u_10m' in icon_data and 'v_10m' in icon_data:
        u = _interpolate_to_points(icon_data['u_10m'], lons, lats)
        v = _interpolate_to_points(icon_data['v_10m'], lons, lats)
        wind = np.sqrt(u**2 + v**2)
        stations_gdf['icon_wind'] = wind
        valid = (~np.isnan(wind)).sum()
        print(f"[ICON] Added icon_wind to {valid}/{len(stations_gdf)} stations")
    
    if 'hsurf' in icon_data:
        vals = _interpolate_to_points(icon_data['hsurf'], lons, lats)
        stations_gdf['icon_hsurf'] = vals
        valid = (~np.isnan(vals)).sum()
        print(f"[ICON] Added icon_hsurf (model orography) to {valid}/{len(stations_gdf)} stations")

    if 't_850' in icon_data:
        vals = _interpolate_to_points(icon_data['t_850'], lons, lats)
        stations_gdf['icon_t850'] = vals
        valid = (~np.isnan(vals)).sum()
        print(f"[ICON] Added icon_t850 (850 hPa) to {valid}/{len(stations_gdf)} stations")

    return stations_gdf

def get_icon_grid(grid_x: np.ndarray, grid_y: np.ndarray,
                  target_time: Optional[datetime] = None) -> Optional[Dict[str, np.ndarray]]:
    """
    Get ICON-EU data interpolated to prediction grid.
    
    Args:
    - grid_x: 1D array of X coordinates (EPSG:2180).
    - grid_y: 1D array of Y coordinates (EPSG:2180).
    - target_time: Locked forecast time.
    
    Returns:
    - Dict with 't2m', 'cloud', 'wind' 2D grids, or None if failed.
    """
    icon_data = fetch_icon_data(target_time)
    if icon_data is None:
        return None
    
    result = {}
    
    if 't_2m' in icon_data:
        result['t2m'] = _interpolate_to_grid(icon_data['t_2m'], grid_x, grid_y)
    
    if 'clct' in icon_data:
        result['cloud'] = np.clip(_interpolate_to_grid(icon_data['clct'], grid_x, grid_y), 0.0, 1.0)
    
    if 'u_10m' in icon_data and 'v_10m' in icon_data:
        u = _interpolate_to_grid(icon_data['u_10m'], grid_x, grid_y)
        v = _interpolate_to_grid(icon_data['v_10m'], grid_x, grid_y)
        result['wind'] = np.sqrt(u**2 + v**2)
    
    if 'hsurf' in icon_data:
        result['hsurf'] = _interpolate_to_grid(icon_data['hsurf'], grid_x, grid_y)

    if 't_850' in icon_data:
        result['t850'] = _interpolate_to_grid(icon_data['t_850'], grid_x, grid_y)

    return result
