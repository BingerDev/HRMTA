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

# Voivodeship name mapping (dual-language support)
VOIVODESHIP_NAMES = {
    # Polish names
    "dolnośląskie": "dolnośląskie",
    "kujawsko-pomorskie": "kujawsko-pomorskie",
    "łódzkie": "łódzkie",
    "lubelskie": "lubelskie",
    "lubuskie": "lubuskie",
    "małopolskie": "małopolskie",
    "mazowieckie": "mazowieckie",
    "opolskie": "opolskie",
    "podkarpackie": "podkarpackie",
    "podlaskie": "podlaskie",
    "pomorskie": "pomorskie",
    "śląskie": "śląskie",
    "świętokrzyskie": "świętokrzyskie",
    "warmińsko-mazurskie": "warmińsko-mazurskie",
    "wielkopolskie": "wielkopolskie",
    "zachodniopomorskie": "zachodniopomorskie",
    # English names
    "lower silesian": "dolnośląskie",
    "kuyavian-pomeranian": "kujawsko-pomorskie",
    "lodz": "łódzkie",
    "lublin": "lubelskie",
    "lubusz": "lubuskie",
    "lesser poland": "małopolskie",
    "masovian": "mazowieckie",
    "opole": "opolskie",
    "subcarpathian": "podkarpackie",
    "podlaskie": "podlaskie",
    "pomeranian": "pomorskie",
    "silesian": "śląskie",
    "holy cross": "świętokrzyskie",
    "warmian-masurian": "warmińsko-mazurskie",
    "greater poland": "wielkopolskie",
    "west pomeranian": "zachodniopomorskie",
}

def is_national_mode(region_name: str) -> bool:
    """Check if region is entire Poland."""
    return region_name.lower() in ("poland", "polska")

def load_region_boundary(region_name: str, crs: str = CRS_WGS84) -> gpd.GeoDataFrame:
    """Load boundary for a specific region."""
    if is_national_mode(region_name):
        return load_poland_boundary(crs)
    
    from .config import VOIVODESHIP_SHAPEFILE
    if not VOIVODESHIP_SHAPEFILE.exists():
        raise FileNotFoundError(f"Voivodeship shapefile not found: {VOIVODESHIP_SHAPEFILE}")
    
    # Normalize region name
    region_lower = region_name.lower()
    canonical_name = VOIVODESHIP_NAMES.get(region_lower)
    
    if canonical_name is None:
        available = sorted(set(VOIVODESHIP_NAMES.values()))
        raise ValueError(f"Region '{region_name}' not found. Available: {available}")
    
    voivodeships = gpd.read_file(VOIVODESHIP_SHAPEFILE)
    match = voivodeships[voivodeships['NAME_1'].str.lower() == canonical_name]
    
    if len(match) == 0:
        raise ValueError(f"Region '{canonical_name}' not found in shapefile.")
    
    if match.crs is None:
        match = match.set_crs(CRS_WGS84)
    if match.crs.to_string() != crs:
        match = match.to_crs(crs)
    
    return match

def get_region_display_name(region_name: str) -> str:
    """Get proper display name for a region."""
    if is_national_mode(region_name):
        return "Polska"
    
    region_lower = region_name.lower()
    canonical = VOIVODESHIP_NAMES.get(region_lower, region_lower)
    return canonical.title()

def get_active_geometry(buffered: bool = False):
    """Get the active region geometry for the current run."""
    from .config import INTERPOLATION_REGION, REGIONAL_BUFFER_KM
    
    region_gdf = load_region_boundary(INTERPOLATION_REGION, CRS_POLAND)
    geom = region_gdf.unary_union
    
    if buffered and not is_national_mode(INTERPOLATION_REGION):
        geom = geom.buffer(REGIONAL_BUFFER_KM * 1000)  # km to meters
    
    return geom

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
        smoothed_data = gaussian_filter(filled_data, sigma, mode='nearest')
        smoothed_mask = gaussian_filter(mask, sigma, mode='constant', cval=0)
    
    # Normalize, divide smoothed data by smoothed weights
    with np.errstate(invalid='ignore', divide='ignore'):
        output = smoothed_data / smoothed_mask
        
    # Restore original NaNs to keep sharp raster borders
    output[mask == 0] = np.nan
    
    return output