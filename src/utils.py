"""
Utility functions for geocoding, cleaning, and geometric operations.
"""
import json
import time
import random
import re
from pathlib import Path
from typing import Tuple, Dict, Optional
import warnings

import geopandas as gpd
from shapely.geometry import Point
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from unidecode import unidecode
from scipy.ndimage import gaussian_filter
import numpy as np

from .config import SHAPEFILE, CRS_WGS84, CRS_POLAND, GEOCACHE_FILE

warnings.filterwarnings("ignore")

# Poland boundary
def load_poland_boundary(crs: str = CRS_WGS84) -> gpd.GeoDataFrame:
    """Load Poland shapefile and return as GeoDataFrame."""
    if not SHAPEFILE.exists():
        raise FileNotFoundError(f"Shapefile not found: {SHAPEFILE}")
    
    gdf = gpd.read_file(SHAPEFILE)
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_POLAND)
    
    if gdf.crs.to_string() != crs:
        gdf = gdf.to_crs(crs)
    
    return gdf

# Global boundary objects
PL_BOUNDARY_WGS84 = load_poland_boundary(CRS_WGS84)
PL_BOUNDARY_2180 = load_poland_boundary(CRS_POLAND)
PL_GEOMETRY_2180 = PL_BOUNDARY_2180.unary_union

def is_in_poland(lat: float, lon: float, tolerance: float = 0) -> bool:
    """Check if point is inside Poland."""
    pt_2180 = (
        gpd.GeoSeries([Point(lon, lat)], crs=CRS_WGS84)
           .to_crs(CRS_POLAND)
           .geometry[0]
    )
    
    poly = PL_GEOMETRY_2180.buffer(tolerance) if tolerance else PL_GEOMETRY_2180
    return poly.contains(pt_2180)

# Geocoding cache
class GeocodingCache:
    """Simple JSON-based geocoding cache."""
    
    def __init__(self, cache_file: Path = GEOCACHE_FILE):
        self.cache_file = cache_file
        self.cache: Dict[str, Tuple[float, float]] = {}
        self.load()
    
    def load(self):
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert lists back to tuples
                    self.cache = {k: tuple(v) for k, v in data.items()}
            except Exception as e:
                print(f"⚠️  Cache load failed: {e}")
                self.cache = {}
    
    def save(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"⚠️  Cache save failed: {e}")
    
    def get(self, key: str) -> Optional[Tuple[float, float]]:
        """Get coordinates from cache."""
        return self.cache.get(key)
    
    def set(self, key: str, coords: Tuple[float, float]):
        """Store coordinates in cache."""
        self.cache[key] = coords
        self.save()

# Global cache instance
_GEOCACHE = GeocodingCache()

# Geocoding
_GEOLOCATOR_RAW = Nominatim(user_agent="HRMTA/2.0", timeout=10)
_GEOLOCATOR = RateLimiter(_GEOLOCATOR_RAW.geocode, min_delay_seconds=1.1)

def geocode_station(
    station_name: str,
    province: Optional[str] = None,
    max_retries: int = 3,
    debug: bool = False
) -> Tuple[Optional[Tuple[float, float]], str]:
    """
    Geocode station name to (lat, lon)
    
    Returns:
        (lat, lon) or None, status_code
        status_code: 'OK', 'NOT_FOUND', 'OUT_OF_POLAND', 'TIMEOUT', 'ERROR'
    """
    # Check cache first
    cache_key = f"{station_name}|{province or 'PL'}"
    cached = _GEOCACHE.get(cache_key)
    if cached is not None:
        if debug:
            print(f"[CACHE] {station_name} -> {cached}")
        return cached, "OK"
    
    # Build query variants
    suffix = ", Poland" if province is None else f", {province}, Poland"
    base = station_name.strip()
    ascii_name = unidecode(base)
    
    queries = [
        base + suffix,
        ascii_name + suffix,
        base + " stacja meteo" + suffix,
        ascii_name + " weather station" + suffix,
    ]
    
    last_error = None
    for query in queries:
        for attempt in range(1, max_retries + 1):
            try:
                location = _GEOLOCATOR(query)
                if location is None:
                    break  # try next query variant
                
                # Check if in Poland
                if is_in_poland(location.latitude, location.longitude):
                    coords = (location.latitude, location.longitude)
                    _GEOCACHE.set(cache_key, coords)
                    if debug:
                        print(f"[OK] {station_name:30s} -> {coords}")
                    return coords, "OK"
                else:
                    if debug:
                        print(f"[OUT] {station_name} -> outside Poland")
                    return None, "OUT_OF_POLAND"
            
            except (GeocoderTimedOut, GeocoderUnavailable):
                last_error = "TIMEOUT"
                if debug:
                    print(f"[TIMEOUT] {station_name} (attempt {attempt}/{max_retries})")
                time.sleep(random.uniform(1, 2))
            
            except Exception as e:
                last_error = "ERROR"
                if debug:
                    print(f"[ERROR] {station_name} -> {e}")
                break
    
    return None, (last_error or "NOT_FOUND")

# Data cleaning
def clean_temperature(temp) -> Optional[float]:
    """Clean temperature value (handle strings, arrows, etc.)."""
    if temp is None:
        return None
    
    if isinstance(temp, (int, float)):
        return float(temp)
    
    # String cleanup
    cleaned = re.sub(r'[↓↑\s,]', '', str(temp))
    cleaned = cleaned.replace(',', '.')
    
    try:
        return float(cleaned)
    except ValueError:
        return None

def clean_station_name(name: str) -> str:
    """Remove keywords like 'min', 'max', 'średnia' from station names."""
    keywords = ["min", "max", "średnia", "avg", "average", "z pomiarów"]
    pattern = re.compile("|".join(keywords), re.IGNORECASE)
    return pattern.sub("", name).strip()

def nan_gaussian_filter(data, sigma):
    """
    Applies Gaussian smoothing handling NaNs intelligently.
    """
    if sigma <= 0:
        return data
        
    # Create a mask of valid data (1 where valid, 0 where NaN)
    mask = np.isfinite(data).astype(float)
    
    # Create a copy of data, replacing NaNs with 0 for convolution
    filled_data = data.copy()
    filled_data[np.isnan(filled_data)] = 0
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        smoothed_data = gaussian_filter(filled_data, sigma, mode='constant', cval=0)
        smoothed_mask = gaussian_filter(mask, sigma, mode='constant', cval=0)
    
    # Normalize, divide smoothed data by smoothed weights
    with np.errstate(invalid='ignore', divide='ignore'):
        output = smoothed_data / smoothed_mask
        
    # Restore original NaNs to keep sharp raster borders
    output[mask == 0] = np.nan
    
    return output