"""
Fetch NWP data from HARMONIE-DMI.
"""
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import requests
import xarray as xr
import geopandas as gpd
from scipy.interpolate import RegularGridInterpolator

from .config import CACHE_DIR, CRS_WGS84, CRS_POLAND, NWP_CACHE_MAX_FILES

# HARMONIE-DMI Configuration
HARMONIE_CONFIG = {
    "stac_endpoint": "https://opendataapi.dmi.dk/v1/forecastdata/collections/harmonie_dini_sf/items",
    "s3_bucket": "https://dmi-opendata.s3.eu-north-1.amazonaws.com/forecastdata/HARMONIE_DINI_SF",
    "southpole_lat": -40.0,
    "southpole_lon": 26.5,
    "cache_hours": 3,  # Cache NWP data for this many hours
}

# Poland bounding box (with buffer for edge coverage)
POLAND_BBOX = {
    "lon_min": 13.5,
    "lon_max": 25.0,
    "lat_min": 48.5,
    "lat_max": 55.5,
}

# NWP cache
_NWP_CACHE: Dict[str, Any] = {}

# Cached Delaunay triangulation for HARMONIE source grid.
_HARMONIE_DELAUNAY = None

def _interpolate_harmonie(src_points: np.ndarray, src_values: np.ndarray,
                          target_coords: np.ndarray) -> np.ndarray:
    """Interpolate HARMONIE data using a cached Delaunay triangulation.

    Uses CloughTocher (cubic) interpolation for C1-continuous output.
    This eliminates gradient discontinuities at grid nodes that cause
    visible rectangular block artifacts when LightGBM splits on NWP features.
    """
    global _HARMONIE_DELAUNAY
    from scipy.spatial import Delaunay
    from scipy.interpolate import CloughTocher2DInterpolator

    # Build or reuse the triangulation
    if (_HARMONIE_DELAUNAY is None or
            len(_HARMONIE_DELAUNAY.points) != len(src_points)):
        _HARMONIE_DELAUNAY = Delaunay(src_points)

    interp = CloughTocher2DInterpolator(_HARMONIE_DELAUNAY, src_values)
    return interp(target_coords)

def rotated_to_regular(rot_lat: np.ndarray, rot_lon: np.ndarray,
                       southpole_lat: float = -40.0, 
                       southpole_lon: float = 26.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert rotated lat/lon to regular WGS84 coordinates.
    
    Args:
    - rot_lat: Rotated latitude array.
    - rot_lon: Rotated longitude array.
    - southpole_lat: Latitude of rotated south pole.
    - southpole_lon: Longitude of rotated south pole.
    
    Returns:
    - Tuple of (regular_lat, regular_lon) arrays.
    """
    to_rad = np.pi / 180.0
    to_deg = 180.0 / np.pi
    
    sin_y_cen = np.sin(to_rad * (southpole_lat + 90.0))
    cos_y_cen = np.cos(to_rad * (southpole_lat + 90.0))
    
    sin_x_rot = np.sin(to_rad * rot_lon)
    cos_x_rot = np.cos(to_rad * rot_lon)
    sin_y_rot = np.sin(to_rad * rot_lat)
    cos_y_rot = np.cos(to_rad * rot_lat)
    
    sin_y_reg = cos_y_cen * sin_y_rot + sin_y_cen * cos_y_rot * cos_x_rot
    sin_y_reg = np.clip(sin_y_reg, -1.0, 1.0)
    
    reg_lat = to_deg * np.arcsin(sin_y_reg)
    cos_y_reg = np.cos(reg_lat * to_rad)
    
    # Avoid division by zero at poles
    cos_y_reg = np.where(np.abs(cos_y_reg) < 1e-10, 1e-10, cos_y_reg)
    
    cos_lon_rad = (cos_y_cen * cos_y_rot * cos_x_rot - sin_y_cen * sin_y_rot) / cos_y_reg
    cos_lon_rad = np.clip(cos_lon_rad, -1.0, 1.0)
    
    sin_lon_rad = cos_y_rot * sin_x_rot / cos_y_reg
    lon_rad = np.arccos(cos_lon_rad)
    lon_rad = np.where(sin_lon_rad < 0.0, -lon_rad, lon_rad)
    
    reg_lon = to_deg * lon_rad + southpole_lon
    
    return reg_lat, reg_lon

def regular_to_rotated(reg_lat: np.ndarray, reg_lon: np.ndarray,
                       southpole_lat: float = -40.0,
                       southpole_lon: float = 26.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert regular WGS84 coordinates to rotated lat/lon.
    
    Inverse of rotated_to_regular(). Used to find grid indices
    for a given geographic location.
    """
    to_rad = np.pi / 180.0
    to_deg = 180.0 / np.pi
    
    # Shift longitude relative to south pole
    lon_shifted = (reg_lon - southpole_lon) * to_rad
    lat_rad = reg_lat * to_rad
    
    sin_y_cen = np.sin(to_rad * (southpole_lat + 90.0))
    cos_y_cen = np.cos(to_rad * (southpole_lat + 90.0))
    
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_shifted)
    cos_lon = np.cos(lon_shifted)
    
    # Rotated latitude
    sin_rot_lat = cos_y_cen * sin_lat - sin_y_cen * cos_lat * cos_lon
    sin_rot_lat = np.clip(sin_rot_lat, -1.0, 1.0)
    rot_lat = to_deg * np.arcsin(sin_rot_lat)
    
    # Rotated longitude
    cos_rot_lat = np.cos(rot_lat * to_rad)
    cos_rot_lat = np.where(np.abs(cos_rot_lat) < 1e-10, 1e-10, cos_rot_lat)
    
    cos_rot_lon = (cos_y_cen * cos_lat * cos_lon + sin_y_cen * sin_lat) / cos_rot_lat
    cos_rot_lon = np.clip(cos_rot_lon, -1.0, 1.0)
    
    sin_rot_lon = cos_lat * sin_lon / cos_rot_lat
    rot_lon = np.arccos(cos_rot_lon) * to_deg
    rot_lon = np.where(sin_rot_lon < 0.0, -rot_lon, rot_lon)
    
    return rot_lat, rot_lon

def _get_latest_model_run() -> Tuple[str, str]:
    """
    Find the latest available HARMONIE model run.
    
    Returns:
        Tuple of (model_run_time, target_time) as ISO strings
    """
    now = datetime.now(timezone.utc)
    
    # Model runs every 3 hours: 00, 03, 06, 09, 12, 15, 18, 21
    # Find latest run that would be available (add ~2-3h processing delay)
    hours_since_midnight = now.hour
    latest_run_hour = (hours_since_midnight // 3) * 3
    
    # Go back one run to ensure data is available
    if hours_since_midnight - latest_run_hour < 2:
        latest_run_hour = (latest_run_hour - 3) % 24
        if latest_run_hour > hours_since_midnight:
            now = now - timedelta(days=1)
    
    model_run = now.replace(hour=latest_run_hour, minute=0, second=0, microsecond=0)
    
    # Target the current hour's forecast
    target_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    
    return (
        model_run.strftime("%Y-%m-%dT%H%M%SZ"),
        target_hour.strftime("%Y-%m-%dT%H:%M:%SZ")
    )

def _fetch_grib_file(target_datetime: Optional[datetime] = None) -> Optional[Path]:
    """
    Fetch the HARMONIE GRIB file with forecast valid time closest to target.
    
    Args:
    - target_datetime: Target forecast valid time, or None for current time.
    
    Returns:
    - Path to downloaded GRIB file, or None if failed.
    """
    cache_dir = CACHE_DIR / "nwp"
    cache_dir.mkdir(exist_ok=True, parents=True)
    
    # Default to current time
    if target_datetime is None:
        target_datetime = datetime.now(timezone.utc)
    
    # Round to nearest hour
    target_datetime = target_datetime.replace(minute=0, second=0, microsecond=0)
    target_str = target_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    try:
        # Query STAC API with datetime filter for exact hour
        params = {
            "limit": 10,
            "datetime": target_str
        }
        
        print(f"[NWP] Searching for forecast at {target_str}...")
        response = requests.get(HARMONIE_CONFIG["stac_endpoint"], params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        features = data.get("features", [])
        if not features:
            # Fallback: try previous hour
            fallback_time = target_datetime - timedelta(hours=1)
            params["datetime"] = fallback_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            print(f"[NWP] No forecast for current hour, trying {params['datetime']}...")
            response = requests.get(HARMONIE_CONFIG["stac_endpoint"], params=params, timeout=30)
            data = response.json()
            features = data.get("features", [])
        
        if not features:
            print("[NWP] ⚠️ No HARMONIE files available for current time")
            return None
        
        def _model_run_time(feature):
            """Parse modelRun into a timezone-aware datetime for sorting."""
            mr = feature.get("properties", {}).get("modelRun", "")
            try:
                return datetime.fromisoformat(mr.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                # Push unparseable entries to the back
                return datetime.min.replace(tzinfo=timezone.utc)

        features.sort(key=_model_run_time, reverse=True)

        best_feature = features[0]

        # Log candidate pool so future debugging is trivial
        newest_run = _model_run_time(features[0])
        oldest_run = _model_run_time(features[-1])
        print(f"[NWP] {len(features)} candidate runs found "
              f"(newest: {newest_run.strftime('%Y-%m-%dT%H:%MZ')}, "
              f"oldest: {oldest_run.strftime('%Y-%m-%dT%H:%MZ')})")
        
        best_href = best_feature["asset"]["data"]["href"]
        file_id = best_feature["id"]
        model_run = best_feature["properties"].get("modelRun", "unknown")
        forecast_time = best_feature["properties"].get("datetime", "unknown")
        
        # Calculate how old/new the forecast is
        try:
            fc_time = datetime.fromisoformat(forecast_time.replace("Z", "+00:00"))
            time_offset = (fc_time - target_datetime).total_seconds() / 3600
            offset_str = f"{time_offset:+.1f}h" if time_offset != 0 else "now"
        except:
            offset_str = "?"
        
        print(f"[NWP] Found: {file_id}")
        print(f"[NWP] Model run: {model_run}, Forecast: {forecast_time} ({offset_str} from now)")
        
        # Check cache
        cache_file_nc = cache_dir / (file_id + '.nc')
        cache_file_grib = cache_dir / file_id
        
        if cache_file_nc.exists():
            print(f"[NWP] Using cached cropped file")
            return cache_file_nc
        if cache_file_grib.exists():
            print(f"[NWP] Found cached raw GRIB, cropping...")
            cropped = _crop_grib_to_poland(cache_file_grib)
            # Delete the raw GRIB if cropping succeeded
            if cropped != cache_file_grib and cropped.exists():
                try:
                    cache_file_grib.unlink()
                except:
                    pass
            return cropped
        
        # Download full GRIB file to a temporary path
        print(f"[NWP] Downloading GRIB file...")
        grib_response = requests.get(best_href, timeout=300, stream=True)
        grib_response.raise_for_status()
        
        with open(cache_file_grib, 'wb') as f:
            for chunk in grib_response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        file_size_mb = cache_file_grib.stat().st_size / (1024 * 1024)
        print(f"[NWP] Downloaded: {file_size_mb:.1f} MB")
        
        # Crop to Poland bbox and save compact NetCDF
        cropped_path = _crop_grib_to_poland(cache_file_grib)
        
        # Delete the raw GRIB to save disk space (~560 MB savings)
        if cropped_path != cache_file_grib and cropped_path.exists():
            try:
                cache_file_grib.unlink()
                print(f"[NWP] Deleted raw GRIB ({file_size_mb:.0f} MB saved)")
            except:
                pass
        
        # Clean old cache files
        if NWP_CACHE_MAX_FILES > 0:
            _clean_cache(cache_dir, max_files=NWP_CACHE_MAX_FILES)
        
        return cropped_path
        
    except Exception as e:
        print(f"[NWP] ❌ Failed to fetch GRIB: {e}")
        return None

def _clean_cache(cache_dir: Path, max_files: int = 5):
    """Remove old cache files, keeping only the most recent."""
    # Clean both .grib and .nc files (cropped Poland subsets)
    for pattern in ["*.grib", "*.nc"]:
        files = sorted(cache_dir.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
        for old_file in files[max_files:]:
            try:
                old_file.unlink()
            except:
                pass

def _crop_grib_to_poland(grib_path: Path) -> Optional[Path]:
    """
    Crop a full-domain HARMONIE GRIB file to the Poland bounding box.
    
    Downloads are ~590 MB (full Northwestern Europe domain). This function:
    1. Opens the GRIB with cfgrib
    2. Converts Poland bbox to rotated-pole indices
    3. Slices all variables to just the Poland region
    4. Saves as a compact NetCDF (~20-30 MB)
    5. Returns the path to the cropped file
    
    The cropped file retains all variables (t2m, cloud, wind, etc.).
    """
    cropped_path = grib_path.with_suffix('.nc')
    if cropped_path.exists():
        return cropped_path
    
    try:
        import cfgrib
        
        datasets = cfgrib.open_datasets(str(grib_path))
        if not datasets:
            print("[NWP] ⚠️ No datasets found in GRIB for cropping")
            return grib_path  # Fall back to full file
                
        # Buffer for edge interpolation (same as used in fetch_nwp_temperature)
        buf = 2.0
        lat_min = POLAND_BBOX["lat_min"] - buf  # 46.5
        lat_max = POLAND_BBOX["lat_max"] + buf  # 57.5
        lon_min = POLAND_BBOX["lon_min"] - buf  # 11.5
        lon_max = POLAND_BBOX["lon_max"] + buf  # 27.0
        
        # Find a dataset with 2D coordinates
        ref_ds = datasets[0]
        
        # Get coordinate arrays
        lat_2d = lon_2d = None
        for coord_name in ['latitude', 'lat', 'y']:
            if coord_name in ref_ds.coords:
                lat_2d = ref_ds.coords[coord_name].values
                break
        for coord_name in ['longitude', 'lon', 'x']:
            if coord_name in ref_ds.coords:
                lon_2d = ref_ds.coords[coord_name].values
                break
        
        if lat_2d is None or lon_2d is None:
            print("[NWP] ⚠️ Cannot find coordinates for cropping, keeping full GRIB")
            return grib_path
        
        # Handle 0-360 longitude range
        if lon_2d.max() > 180:
            lon_2d = np.where(lon_2d > 180, lon_2d - 360, lon_2d)
        
        # For 1D coords, meshgrid them
        if lat_2d.ndim == 1 and lon_2d.ndim == 1:
            lon_2d_full, lat_2d_full = np.meshgrid(lon_2d, lat_2d)
        else:
            lat_2d_full = lat_2d
            lon_2d_full = lon_2d
        
        # Build a 2D mask of grid points within the Poland bbox
        poland_mask_2d = (
            (lat_2d_full >= lat_min) & (lat_2d_full <= lat_max) &
            (lon_2d_full >= lon_min) & (lon_2d_full <= lon_max)
        )
        
        if not poland_mask_2d.any():
            print("[NWP] ⚠️ No grid points found in Poland bbox, keeping full GRIB")
            return grib_path
        
        # Find the bounding rows/cols of the masked region
        rows_with_data = np.any(poland_mask_2d, axis=1)
        cols_with_data = np.any(poland_mask_2d, axis=0)
        
        row_start = np.argmax(rows_with_data)
        row_end = len(rows_with_data) - np.argmax(rows_with_data[::-1])
        col_start = np.argmax(cols_with_data)
        col_end = len(cols_with_data) - np.argmax(cols_with_data[::-1])
        
        orig_shape = lat_2d_full.shape
        crop_shape = (row_end - row_start, col_end - col_start)
        reduction = (1 - (crop_shape[0] * crop_shape[1]) / (orig_shape[0] * orig_shape[1])) * 100
        
        print(f"[NWP] Cropping: {orig_shape} → {crop_shape} ({reduction:.0f}% reduction)")
        
        cropped_vars = {}
        
        # Determine primary coordinate dimension names from the first dataset
        for ds in datasets:
            for var_name in ds.data_vars:
                arr = ds[var_name].values.squeeze()
                if arr.ndim < 2:
                    continue
                # Crop the last two dimensions (y, x)
                if arr.ndim == 2:
                    cropped_vars[var_name] = arr[row_start:row_end, col_start:col_end]
                elif arr.ndim == 3:
                    cropped_vars[var_name] = arr[:, row_start:row_end, col_start:col_end]
                
                # Preserve GRIB attributes for variable identification
                if var_name not in cropped_vars:
                    continue
                attrs = ds[var_name].attrs
                cropped_vars[f"_attrs_{var_name}"] = {
                    'GRIB_shortName': attrs.get('GRIB_shortName', ''),
                    'GRIB_name': attrs.get('GRIB_name', ''),
                    'GRIB_typeOfLevel': attrs.get('GRIB_typeOfLevel', ''),
                    'GRIB_units': attrs.get('GRIB_units', attrs.get('units', '')),
                    'units': attrs.get('units', ''),
                }
            
            # Also crop coordinate arrays from this dataset
            for coord_name in ds.coords:
                if coord_name in cropped_vars:
                    continue
                coord_vals = ds.coords[coord_name].values
                if coord_vals.ndim == 2 and coord_vals.shape == orig_shape:
                    cropped_vars[coord_name] = coord_vals[row_start:row_end, col_start:col_end]
                elif coord_vals.ndim == 1:
                    # 1D coord: slice on the appropriate axis
                    if len(coord_vals) == orig_shape[0]:  # y-axis
                        cropped_vars[coord_name] = coord_vals[row_start:row_end]
                    elif len(coord_vals) == orig_shape[1]:  # x-axis
                        cropped_vars[coord_name] = coord_vals[col_start:col_end]
                    # Other 1D coords (time, step, number) keep as-is
                    elif coord_name == 'heightAboveGround':
                        cropped_vars[coord_name] = coord_vals
        
        # Extract T850 (850 hPa temperature)
        if 't850' not in cropped_vars:
            for ds in datasets:
                if 'isobaricInhPa' not in ds.coords:
                    continue
                pressure_levels = np.atleast_1d(ds.coords['isobaricInhPa'].values)
                # 850 hPa may be stored in hPa or Pa
                target_lvl = None
                if 850.0 in pressure_levels or 850 in pressure_levels:
                    target_lvl = 850
                elif 85000.0 in pressure_levels or 85000 in pressure_levels:
                    target_lvl = 85000
                if target_lvl is None:
                    continue
                for var_name in ds.data_vars:
                    attrs = ds[var_name].attrs
                    short = attrs.get('GRIB_shortName', '')
                    units = attrs.get('units', attrs.get('GRIB_units', ''))
                    if short == 't' or units == 'K':
                        try:
                            t850_raw = ds[var_name].sel(isobaricInhPa=target_lvl).values
                            t850_crop = t850_raw[row_start:row_end, col_start:col_end]
                            if np.nanmean(t850_crop) > 100:
                                t850_crop = t850_crop - 273.15
                            cropped_vars['t850'] = t850_crop
                            cropped_vars['_attrs_t850'] = {
                                'GRIB_shortName': 't',
                                'GRIB_name': 'Temperature at 850 hPa',
                                'GRIB_typeOfLevel': 'isobaricInhPa',
                                'units': '°C',
                            }
                            print(f"[NWP] ✓ Extracted T850: "
                                  f"{np.nanmin(t850_crop):.1f} to {np.nanmax(t850_crop):.1f}°C")
                        except (KeyError, ValueError) as e:
                            print(f"[NWP] ⚠ T850 extraction failed: {e}")
                        break
                if 't850' in cropped_vars:
                    break

        # Save as NetCDF using xarray
        ds_out = xr.Dataset()
        attrs_store = {}
        
        for key, val in cropped_vars.items():
            if key.startswith('_attrs_'):
                attrs_store[key.replace('_attrs_', '')] = val
                continue
            if isinstance(val, np.ndarray):
                if val.ndim == 2:
                    ds_out[key] = xr.DataArray(val, dims=['y', 'x'])
                elif val.ndim == 3:
                    # Use unique level dimension per variable
                    ds_out[key] = xr.DataArray(val, dims=[f'{key}_level', 'y', 'x'])
                elif val.ndim == 1:
                    # Determine dimension name
                    if len(val) == crop_shape[0]:
                        ds_out[key] = xr.DataArray(val, dims=['y'])
                    elif len(val) == crop_shape[1]:
                        ds_out[key] = xr.DataArray(val, dims=['x'])
                    else:
                        ds_out[key] = xr.DataArray(val, dims=[key])
                elif val.ndim == 0:
                    ds_out[key] = xr.DataArray(val)
        
        # Attach GRIB attributes to variables for downstream identification
        for var_name, attrs in attrs_store.items():
            if var_name in ds_out:
                ds_out[var_name].attrs.update(attrs)
        
        # Store crop metadata for reference
        ds_out.attrs['crop_row_start'] = row_start
        ds_out.attrs['crop_row_end'] = row_end
        ds_out.attrs['crop_col_start'] = col_start
        ds_out.attrs['crop_col_end'] = col_end
        ds_out.attrs['original_shape'] = str(orig_shape)
        ds_out.attrs['source'] = 'HARMONIE-DINI-SF (cropped to Poland)'
        
        ds_out.to_netcdf(cropped_path)
        
        cropped_size = cropped_path.stat().st_size / (1024 * 1024)
        print(f"[NWP] ✓ Cropped to Poland: {cropped_size:.1f} MB")
        
        return cropped_path
        
    except Exception as e:
        print(f"[NWP] ⚠️ GRIB cropping failed ({e}), falling back to full file")
        import traceback
        traceback.print_exc()
        return grib_path

def _parse_grib_temperature(grib_path: Path) -> Optional[xr.DataArray]:
    """
    Parse 2m temperature from HARMONIE GRIB file.
    
    Args:
    - grib_path: Path to GRIB file.
    
    Returns:
    - xarray DataArray with temperature data, or None if failed.
    """
    try:
        # Handle both GRIB (raw) and NetCDF (cropped) files
        if grib_path.suffix == '.nc':
            ds_nc = xr.open_dataset(grib_path)
            for lat_name in ['latitude', 'lat']:
                if lat_name in ds_nc.data_vars:
                    ds_nc = ds_nc.set_coords(lat_name)
                    break
            for lon_name in ['longitude', 'lon']:
                if lon_name in ds_nc.data_vars:
                    ds_nc = ds_nc.set_coords(lon_name)
                    break
            # Wrap in list to match cfgrib.open_datasets format
            datasets = [ds_nc]
            print(f"[NWP] Loading cropped NetCDF ({len(ds_nc.data_vars)} variables)")
        else:
            import cfgrib
            datasets = cfgrib.open_datasets(str(grib_path))
        print(f"[NWP] Scanning {len(datasets)} datasets for 2m temperature...")
        
        # Find t2m at heightAboveGround=2m
        for i, ds in enumerate(datasets):
            if "t2m" in ds.data_vars:
                temp = ds["t2m"]
                
                # If t2m is 3D (has a height/level dimension), select 2m slice
                if temp.ndim > 2:
                    if "heightAboveGround" in ds.coords:
                        height_vals = np.atleast_1d(ds.coords["heightAboveGround"].values).ravel()
                        if 2.0 in height_vals:
                            try:
                                temp = temp.sel(heightAboveGround=2.0)
                            except (KeyError, ValueError):
                                # Level dim may have a different name in cropped NetCDF
                                temp = temp[0]  # Take first level as fallback
                    else:
                        temp = temp[0]  # Take first level
                
                # Verify it's at 2m (if height coord exists and t2m is 2D, trust it)
                if "heightAboveGround" in ds.coords and temp.ndim == 2:
                    height_vals = np.atleast_1d(ds.coords["heightAboveGround"].values).ravel()
                    if 2.0 not in height_vals:
                        continue  # Not at 2m, skip
                
                if float(temp.values.mean()) > 100:  # Kelvin
                    temp = temp - 273.15
                print(f"[NWP] ✓ Found t2m in dataset {i} at heightAboveGround=2m")
                print(f"[NWP] Temperature range: {float(temp.min()):.1f} to {float(temp.max()):.1f}°C")
                return temp
        
        # Find variable with GRIB_shortName="2t"
        for i, ds in enumerate(datasets):
            for var in ds.data_vars:
                attrs = ds[var].attrs
                short_name = attrs.get("GRIB_shortName", "")
                grib_name = attrs.get("GRIB_name", "").lower()
                
                if short_name == "2t" or "2 metre temperature" in grib_name:
                    temp = ds[var]
                    if float(temp.values.mean()) > 100:
                        temp = temp - 273.15
                    print(f"[NWP] ✓ Found 2m temp via GRIB attributes in dataset {i}, var={var}")
                    print(f"[NWP] Temperature range: {float(temp.min()):.1f} to {float(temp.max()):.1f}°C")
                    return temp
        
        # Find 't' variable at heightAboveGround=2m
        for i, ds in enumerate(datasets):
            if "t" in ds.data_vars and "heightAboveGround" in ds.coords:
                heights = ds.coords["heightAboveGround"].values
                
                # Handle scalar or array heights
                if np.isscalar(heights) or heights.ndim == 0:
                    if float(heights) == 2.0:
                        temp = ds["t"]
                        if float(temp.values.mean()) > 100:
                            temp = temp - 273.15
                        print(f"[NWP] ✓ Found 't' at 2m in dataset {i}")
                        return temp
                elif 2.0 in heights:
                    temp = ds["t"].sel(heightAboveGround=2.0)
                    if float(temp.values.mean()) > 100:
                        temp = temp - 273.15
                    print(f"[NWP] ✓ Found 't' at 2m in dataset {i}")
                    return temp
        
        # Find any temperature-like variable at surface/2m
        excluded_levels = ["cloudTop", "isobaricInhPa", "hybrid"]
        
        for i, ds in enumerate(datasets):
            # Skip datasets with cloud/upper atmosphere levels
            has_excluded = any(lvl in ds.coords for lvl in excluded_levels)
            if has_excluded:
                continue
            
            for var in ds.data_vars:
                attrs = ds[var].attrs
                units = attrs.get("units", attrs.get("GRIB_units", ""))
                
                # Look for temperature in Kelvin at surface level
                if units == "K" and "heightAboveGround" in ds.coords:
                    height = ds.coords["heightAboveGround"].values
                    if np.isscalar(height) or height.ndim == 0:
                        height = float(height)
                    else:
                        height = float(height[0])
                    
                    # Only accept low-level (surface) temperatures
                    if height <= 10:
                        temp = ds[var]
                        temp_mean = float(temp.values[~np.isnan(temp.values)].mean())
                        
                        # Sanity check: surface temp should be roughly 220-320 K
                        if 220 < temp_mean < 320:
                            if temp_mean > 100:
                                temp = temp - 273.15
                            print(f"[NWP] ✓ Found surface temp in dataset {i}, var={var}, height={height}m")
                            print(f"[NWP] Temperature range: {float(temp.min()):.1f} to {float(temp.max()):.1f}°C")
                            return temp
        
        print("[NWP] ⚠️ Could not find 2m temperature in any dataset")
        
        # Debug: list what we found
        print("[NWP] Available temperature-like variables:")
        for i, ds in enumerate(datasets):
            for var in ds.data_vars:
                attrs = ds[var].attrs
                if attrs.get("units") == "K" or attrs.get("GRIB_units") == "K":
                    levels = {k: ds.coords[k].values for k in ds.coords if k not in ['latitude', 'longitude', 'time', 'step', 'valid_time', 'number']}
                    print(f"  Dataset {i}: {var}, levels={levels}")
        
        return None
        
    except Exception as e:
        print(f"[NWP] ❌ Failed to parse GRIB: {e}")
        import traceback
        traceback.print_exc()
        return None

def _parse_grib_t850(grib_path: Path) -> Optional[np.ndarray]:
    """
    Parse 850 hPa temperature from HARMONIE GRIB / cropped NetCDF.

    Returns:
        2D numpy array of T850 in °C (same spatial grid as T2m), or None
    """
    try:
        if grib_path.suffix == '.nc':
            ds_nc = xr.open_dataset(grib_path)
            if 't850' in ds_nc.data_vars:
                t850 = ds_nc['t850'].values.squeeze()
                print(f"[NWP] ✓ T850 from NetCDF: "
                      f"{np.nanmin(t850):.1f} to {np.nanmax(t850):.1f}°C")
                return t850
            return None
        else:
            # Raw GRIB
            import cfgrib
            for ds in cfgrib.open_datasets(str(grib_path)):
                if 'isobaricInhPa' not in ds.coords:
                    continue
                levels = np.atleast_1d(ds.coords['isobaricInhPa'].values)
                target = 850 if (850 in levels or 850.0 in levels) else \
                         85000 if (85000 in levels or 85000.0 in levels) else None
                if target is None:
                    continue
                for var in ds.data_vars:
                    sn = ds[var].attrs.get('GRIB_shortName', '')
                    if sn == 't' or ds[var].attrs.get('units', '') == 'K':
                        t850 = ds[var].sel(isobaricInhPa=target).values
                        if np.nanmean(t850) > 100:
                            t850 = t850 - 273.15
                        print(f"[NWP] ✓ T850 from GRIB: "
                              f"{np.nanmin(t850):.1f} to {np.nanmax(t850):.1f}°C")
                        return t850
            return None
    except Exception as e:
        print(f"[NWP] ⚠ T850 parse failed: {e}")
        return None

def _parse_grib_cloud_wind(grib_path: Path) -> Optional[Dict[str, np.ndarray]]:
    """
    Parse cloud cover and wind speed from HARMONIE GRIB file.
    
    Args:
    - grib_path: Path to GRIB file.
    
    Returns:
    - Dict with 'cloud_cover' and 'wind_speed' arrays, or None if failed.
    """
    try:
        # Handle both GRIB (raw) and NetCDF (cropped) files
        if grib_path.suffix == '.nc':
            ds_nc = xr.open_dataset(grib_path)
            datasets = [ds_nc]
        else:
            import cfgrib
            datasets = cfgrib.open_datasets(str(grib_path))
        result = {}
        
        # Extract cloud cover (HARMONIE uses 'cc' not 'tcc')
        for ds in datasets:
            for var in ds.data_vars:
                attrs = ds[var].attrs
                short_name = attrs.get("GRIB_shortName", "")
                grib_name = attrs.get("GRIB_name", "").lower()
                
                # HARMONIE uses 'cc' for cloud fraction
                if short_name in ["cc", "tcc"] or "cloud cover" in grib_name:
                    cloud = ds[var].values.squeeze()
                    # Handle multi-level cloud (take mean or surface level)
                    if cloud.ndim == 3:
                        cloud = cloud[0]  # Take first level (surface)
                    # Normalize to 0-1 if needed
                    if np.nanmax(cloud) > 1.5:
                        cloud = cloud / 100.0
                    result["cloud_cover"] = cloud
                    print(f"[NWP] ✓ Found cloud cover ({short_name}): {np.nanmin(cloud):.2f} to {np.nanmax(cloud):.2f}")
                    break
            if "cloud_cover" in result:
                break
        
        # Extract 10m wind components
        u_wind = None
        v_wind = None
        
        for ds in datasets:
            for var in ds.data_vars:
                attrs = ds[var].attrs
                short_name = attrs.get("GRIB_shortName", "")
                level_type = attrs.get("GRIB_typeOfLevel", "")
                
                # Look for 10m wind (heightAboveGround level type)
                if level_type == "heightAboveGround" or "10" in short_name:
                    if short_name in ["10u", "u"] and u_wind is None:
                        data = ds[var].values.squeeze()
                        # Ensure 2D (take first level if 3D)
                        if data.ndim == 3:
                            data = data[0]
                        u_wind = data
                    elif short_name in ["10v", "v"] and v_wind is None:
                        data = ds[var].values.squeeze()
                        if data.ndim == 3:
                            data = data[0]
                        v_wind = data
        
        # Compute wind speed from u/v components
        if u_wind is not None and v_wind is not None:
            wind_speed = np.sqrt(u_wind**2 + v_wind**2)
            result["wind_speed"] = wind_speed
            print(f"[NWP] ✓ Found 10m wind: {np.nanmin(wind_speed):.1f} to {np.nanmax(wind_speed):.1f} m/s")
        
        if not result:
            print("[NWP] ⚠️ Could not find cloud cover or wind in GRIB")
            return None
        
        return result
        
    except Exception as e:
        print(f"[NWP] ⚠️ Failed to parse cloud/wind from GRIB: {e}")
        return None

def fetch_nwp_temperature(target_points: Optional[gpd.GeoDataFrame] = None,
                          grid_x: Optional[np.ndarray] = None,
                          grid_y: Optional[np.ndarray] = None,
                          target_time: Optional[datetime] = None) -> Optional[np.ndarray]:
    """
    Fetch HARMONIE NWP 2m temperature for Poland.
    
    Args:
    - target_points: GeoDataFrame with point geometries (WGS84).
    - grid_x: 1D array of X coordinates (EPSG:2180).
    - grid_y: 1D array of Y coordinates (EPSG:2180).
    - target_time: Locked forecast time (ensures station & grid use same NWP hour).
    
    Returns:
    - Interpolated temperature values (1D for points, 2D for grid).
    """
    # Check cache
    ref_time = target_time or datetime.now(timezone.utc)
    cache_key = ref_time.strftime("%Y%m%d_%H")
    if cache_key in _NWP_CACHE:
        print("[NWP] Using cached NWP data")
        nwp_data = _NWP_CACHE[cache_key]
    else:
        # Fetch and parse GRIB
        grib_path = _fetch_grib_file(target_datetime=ref_time)
        if grib_path is None:
            return None
        
        temp = _parse_grib_temperature(grib_path)
        if temp is None:
            return None
        
        # Get coordinate arrays from GRIB
        print(f"[NWP] Available coords: {list(temp.coords)}")
        
        rot_x = None
        rot_y = None
        
        # Try different coordinate names
        for x_name in ["x", "longitude", "lon"]:
            if x_name in temp.coords:
                rot_x = temp.coords[x_name].values
                break
        for y_name in ["y", "latitude", "lat"]:
            if y_name in temp.coords:
                rot_y = temp.coords[y_name].values
                break
        
        if rot_x is None or rot_y is None:
            print("[NWP] ⚠️ Could not find coordinate arrays")
            return None
        
        # Check if coordinates are 1D or 2D
        print(f"[NWP] Coordinate shapes: x={rot_x.shape}, y={rot_y.shape}")
        
        if rot_x.max() > 180:
            print(f"[NWP] Adjusting longitude from 0-360 to -180/180 range")
            rot_x = np.where(rot_x > 180, rot_x - 360, rot_x)
        
        if rot_x.ndim == 2 and rot_y.ndim == 2:
            reg_lon = rot_x
            reg_lat = rot_y
        elif rot_x.ndim == 1 and rot_y.ndim == 1:
            # Create 2D grid from 1D coords
            reg_lon, reg_lat = np.meshgrid(rot_x, rot_y)
        else:
            print(f"[NWP] ⚠️ Unexpected coordinate dimensions: x.ndim={rot_x.ndim}, y.ndim={rot_y.ndim}")
            return None
        
        print(f"[NWP] WGS84 lat: {reg_lat.min():.2f} to {reg_lat.max():.2f}")
        print(f"[NWP] WGS84 lon: {reg_lon.min():.2f} to {reg_lon.max():.2f}")
        
        # Store 1D versions for interpolation
        if rot_x.ndim == 2:
            lon_1d = rot_x[0, :]  # First row
            lat_1d = rot_y[:, 0]  # First column
        else:
            lon_1d = rot_x
            lat_1d = rot_y
        
        nwp_data = {
            "temp": temp.values.squeeze(),
            "lat": reg_lat,
            "lon": reg_lon,
            "rot_x": lon_1d,  # Store for interpolation
            "rot_y": lat_1d,
        }

        # Try to extract T850 from the same file
        t850 = _parse_grib_t850(grib_path)
        if t850 is not None:
            nwp_data["t850"] = t850

        _NWP_CACHE[cache_key] = nwp_data
        
        # Clear old cache entries
        for old_key in list(_NWP_CACHE.keys()):
            if old_key != cache_key:
                del _NWP_CACHE[old_key]
    
    # Subset to Poland region (with buffer for interpolation)
    buffer = 2.0  # degrees
    lat_mask = (nwp_data["lat"] >= POLAND_BBOX["lat_min"] - buffer) & (nwp_data["lat"] <= POLAND_BBOX["lat_max"] + buffer)
    lon_mask = (nwp_data["lon"] >= POLAND_BBOX["lon_min"] - buffer) & (nwp_data["lon"] <= POLAND_BBOX["lon_max"] + buffer)
    poland_mask = lat_mask & lon_mask
    
    if not poland_mask.any():
        print("[NWP] ⚠️ No NWP data covers Poland")
        return None
    
    # Extract Poland subset for interpolation
    src_lat = nwp_data["lat"][poland_mask]
    src_lon = nwp_data["lon"][poland_mask]
    src_temp = nwp_data["temp"][poland_mask]
    
    src_points = np.column_stack([src_lon, src_lat])  # (lon, lat) for griddata
    print(f"[NWP] Using {len(src_points)} source points for interpolation")
    
    # Interpolate to target points
    if target_points is not None:
        lons = target_points.geometry.x.values
        lats = target_points.geometry.y.values
        target_coords = np.column_stack([lons, lats])
        
        nwp_temps = _interpolate_harmonie(src_points, src_temp, target_coords)

        valid_count = (~np.isnan(nwp_temps)).sum()
        print(f"[NWP] Interpolated to {len(target_points)} points ({valid_count} valid)")
        return nwp_temps

    # Interpolate to grid
    if grid_x is not None and grid_y is not None:
        import pyproj

        # Transform grid from EPSG:2180 to WGS84
        transformer = pyproj.Transformer.from_crs(CRS_POLAND, CRS_WGS84, always_xy=True)
        grid_x_2d, grid_y_2d = np.meshgrid(grid_x, grid_y)
        lon_2d, lat_2d = transformer.transform(grid_x_2d, grid_y_2d)

        # Flatten for griddata
        target_coords = np.column_stack([lon_2d.ravel(), lat_2d.ravel()])

        # Interpolate using cached Delaunay (curvilinear source grid)
        nwp_temps_flat = _interpolate_harmonie(src_points, src_temp, target_coords)
        nwp_temps = nwp_temps_flat.reshape(grid_x_2d.shape)
        
        valid_count = (~np.isnan(nwp_temps)).sum()
        print(f"[NWP] Interpolated to {grid_x_2d.shape} grid ({valid_count} valid points)")
        return nwp_temps
    
    # Return raw Poland-cropped data
    return nwp_data["temp"]

def get_nwp_at_stations(stations_gdf: gpd.GeoDataFrame, target_time: Optional[datetime] = None) -> gpd.GeoDataFrame:
    """
    Add NWP columns to station GeoDataFrame.

    Adds: nwp_temp, nwp_cloud, nwp_wind, nwp_t850 (if available in GRIB).
    
    Args:
    - stations_gdf: GeoDataFrame with station points.
    - target_time: Locked forecast time for temporal consistency.
    
    Returns:
    - GeoDataFrame with NWP columns added.
    """
    stations = stations_gdf.copy()
    
    # Ensure WGS84
    if stations.crs and str(stations.crs) != CRS_WGS84:
        stations = stations.to_crs(CRS_WGS84)
    
    nwp_temps = fetch_nwp_temperature(target_points=stations, target_time=target_time)
    
    if nwp_temps is not None:
        stations_gdf["nwp_temp"] = nwp_temps
        valid_count = (~np.isnan(nwp_temps)).sum()
        print(f"[NWP] Added NWP temperature to {valid_count}/{len(stations_gdf)} stations")
    else:
        stations_gdf["nwp_temp"] = np.nan
        print("[NWP] ⚠️ Could not fetch NWP data, using NaN")
    
    # Extract additional NWP features (cloud cover, wind speed)
    cache_dir = CACHE_DIR / "nwp"
    # Look for cached files
    cached_files = sorted(cache_dir.glob("*.nc"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not cached_files:
        cached_files = sorted(cache_dir.glob("*.grib"), key=lambda f: f.stat().st_mtime, reverse=True)
    
    if cached_files:
        grib_path = cached_files[0]
        cloud_wind = _parse_grib_cloud_wind(grib_path)
        
        if cloud_wind:
            # Get coordinates from cached NWP data
            if _NWP_CACHE:
                cache_key = list(_NWP_CACHE.keys())[-1]  # Most recent entry
                nwp_data = _NWP_CACHE[cache_key]
                
                # Prepare source points for interpolation
                poland_mask = (
                    (nwp_data["lat"] >= POLAND_BBOX["lat_min"] - 2.0) & 
                    (nwp_data["lat"] <= POLAND_BBOX["lat_max"] + 2.0) &
                    (nwp_data["lon"] >= POLAND_BBOX["lon_min"] - 2.0) & 
                    (nwp_data["lon"] <= POLAND_BBOX["lon_max"] + 2.0)
                )
                
                src_lat = nwp_data["lat"][poland_mask]
                src_lon = nwp_data["lon"][poland_mask]
                src_points = np.column_stack([src_lon, src_lat])
                
                # Target coordinates
                lons = stations.geometry.x.values
                lats = stations.geometry.y.values
                target_coords = np.column_stack([lons, lats])
                
                # Interpolate cloud cover (clip for cubic overshoot)
                if "cloud_cover" in cloud_wind:
                    src_cloud = cloud_wind["cloud_cover"][poland_mask]
                    nwp_cloud = np.clip(
                        _interpolate_harmonie(src_points, src_cloud, target_coords), 0.0, 1.0)
                    stations_gdf["nwp_cloud"] = nwp_cloud
                    valid = (~np.isnan(nwp_cloud)).sum()
                    print(f"[NWP] Added cloud cover to {valid}/{len(stations_gdf)} stations")

                # Interpolate wind speed (clip for cubic overshoot)
                if "wind_speed" in cloud_wind:
                    src_wind = cloud_wind["wind_speed"][poland_mask]
                    nwp_wind = np.clip(
                        _interpolate_harmonie(src_points, src_wind, target_coords), 0.0, None)
                    stations_gdf["nwp_wind"] = nwp_wind
                    valid = (~np.isnan(nwp_wind)).sum()
                    print(f"[NWP] Added wind speed to {valid}/{len(stations_gdf)} stations")

                # Interpolate T850 (cached during fetch_nwp_temperature)
                if "t850" in nwp_data:
                    src_t850 = nwp_data["t850"][poland_mask]
                    nwp_t850 = _interpolate_harmonie(src_points, src_t850, target_coords)
                    stations_gdf["nwp_t850"] = nwp_t850
                    valid = (~np.isnan(nwp_t850)).sum()
                    print(f"[NWP] Added T850 to {valid}/{len(stations_gdf)} stations")

    return stations_gdf

def get_nwp_cloud_wind_grid(grid_x: np.ndarray, grid_y: np.ndarray,
                            target_time: Optional[datetime] = None) -> Optional[Dict[str, np.ndarray]]:
    """
    Get NWP cloud cover and wind speed interpolated to prediction grid.

    Args:
    - grid_x: 1D array of X coordinates (EPSG:2180).
    - grid_y: 1D array of Y coordinates (EPSG:2180).
    - target_time: Locked forecast time for temporal consistency.

    Returns:
    - Dict with 'cloud_cover' and 'wind_speed' 2D arrays, or None.
    """
    # Ensure NWP cache is populated
    if not _NWP_CACHE:
        fetch_nwp_temperature(grid_x=grid_x, grid_y=grid_y, target_time=target_time)
    if not _NWP_CACHE:
        return None

    # Get cached NWP coordinate data
    cache_key = list(_NWP_CACHE.keys())[-1]
    nwp_data = _NWP_CACHE[cache_key]

    # Find and parse GRIB cloud/wind
    cache_dir = CACHE_DIR / "nwp"
    cached_files = sorted(cache_dir.glob("*.nc"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not cached_files:
        cached_files = sorted(cache_dir.glob("*.grib"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not cached_files:
        return None

    cloud_wind = _parse_grib_cloud_wind(cached_files[0])
    if not cloud_wind:
        return None

    # Subset to Poland region
    buffer = 2.0
    poland_mask = (
        (nwp_data["lat"] >= POLAND_BBOX["lat_min"] - buffer) &
        (nwp_data["lat"] <= POLAND_BBOX["lat_max"] + buffer) &
        (nwp_data["lon"] >= POLAND_BBOX["lon_min"] - buffer) &
        (nwp_data["lon"] <= POLAND_BBOX["lon_max"] + buffer)
    )
    if not poland_mask.any():
        return None

    src_lat = nwp_data["lat"][poland_mask]
    src_lon = nwp_data["lon"][poland_mask]
    src_points = np.column_stack([src_lon, src_lat])

    # Transform grid from EPSG:2180 to WGS84
    import pyproj

    transformer = pyproj.Transformer.from_crs(CRS_POLAND, CRS_WGS84, always_xy=True)
    grid_x_2d, grid_y_2d = np.meshgrid(grid_x, grid_y)
    lon_2d, lat_2d = transformer.transform(grid_x_2d, grid_y_2d)
    target_coords = np.column_stack([lon_2d.ravel(), lat_2d.ravel()])

    result = {}

    if "cloud_cover" in cloud_wind:
        src_cloud = cloud_wind["cloud_cover"][poland_mask]
        cloud_flat = np.clip(_interpolate_harmonie(src_points, src_cloud, target_coords), 0.0, 1.0)
        result["cloud_cover"] = cloud_flat.reshape(grid_x_2d.shape)
        valid = (~np.isnan(result["cloud_cover"])).sum()
        print(f"[NWP] Grid cloud cover: {valid} valid points")

    if "wind_speed" in cloud_wind:
        src_wind = cloud_wind["wind_speed"][poland_mask]
        wind_flat = np.clip(_interpolate_harmonie(src_points, src_wind, target_coords), 0.0, None)
        result["wind_speed"] = wind_flat.reshape(grid_x_2d.shape)
        valid = (~np.isnan(result["wind_speed"])).sum()
        print(f"[NWP] Grid wind speed: {valid} valid points")

    return result if result else None

def get_nwp_grid(grid_x: np.ndarray, grid_y: np.ndarray, target_time: Optional[datetime] = None) -> Optional[np.ndarray]:
    """
    Get NWP temperature interpolated to prediction grid.
    
    Args:
    - grid_x: 1D array of X coordinates (EPSG:2180).
    - grid_y: 1D array of Y coordinates (EPSG:2180).
    - target_time: Locked forecast time for temporal consistency.
    
    Returns:
    - 2D temperature grid, or None if failed.
    """
    return fetch_nwp_temperature(grid_x=grid_x, grid_y=grid_y, target_time=target_time)

def get_nwp_t850_grid(grid_x: np.ndarray, grid_y: np.ndarray,
                      target_time: Optional[datetime] = None) -> Optional[np.ndarray]:
    """
    Get HARMONIE T850 interpolated to prediction grid.

    Uses the T850 field cached during the temperature fetch.
    Returns a 2D array (same shape as T2m grid), or None if T850 unavailable.
    """
    # Ensure NWP cache is populated
    if not _NWP_CACHE:
        fetch_nwp_temperature(grid_x=grid_x, grid_y=grid_y, target_time=target_time)
    if not _NWP_CACHE:
        return None

    cache_key = list(_NWP_CACHE.keys())[-1]
    nwp_data = _NWP_CACHE[cache_key]

    if "t850" not in nwp_data:
        return None

    # Subset to Poland
    buffer = 2.0
    poland_mask = (
        (nwp_data["lat"] >= POLAND_BBOX["lat_min"] - buffer) &
        (nwp_data["lat"] <= POLAND_BBOX["lat_max"] + buffer) &
        (nwp_data["lon"] >= POLAND_BBOX["lon_min"] - buffer) &
        (nwp_data["lon"] <= POLAND_BBOX["lon_max"] + buffer)
    )
    if not poland_mask.any():
        return None

    src_lat = nwp_data["lat"][poland_mask]
    src_lon = nwp_data["lon"][poland_mask]
    src_t850 = nwp_data["t850"][poland_mask]
    src_points = np.column_stack([src_lon, src_lat])

    import pyproj
    transformer = pyproj.Transformer.from_crs(CRS_POLAND, CRS_WGS84, always_xy=True)
    grid_x_2d, grid_y_2d = np.meshgrid(grid_x, grid_y)
    lon_2d, lat_2d = transformer.transform(grid_x_2d, grid_y_2d)
    target_coords = np.column_stack([lon_2d.ravel(), lat_2d.ravel()])

    t850_flat = _interpolate_harmonie(src_points, src_t850, target_coords)
    t850_grid = t850_flat.reshape(grid_x_2d.shape)

    valid = (~np.isnan(t850_grid)).sum()
    print(f"[NWP] Grid T850: {valid} valid points")
    return t850_grid
