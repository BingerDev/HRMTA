"""
Fetch temperature data from IMGW, Netatmo, TraxElektronik, and Edwin.
"""
import re
import json
import concurrent.futures
from datetime import datetime, timedelta, timezone
from typing import List, Tuple
import requests
import time
import pandas as pd
from bs4 import BeautifulSoup

from .config import (IMGW_PROVINCES, IMGW_DATA_MODE, TRAX_REGION_IDS,
                     NETATMO_CONFIG, EDWIN_CONFIG, PWS_DEDUP_RADIUS_M)
from .utils import is_in_poland, clean_temperature

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# IMGW
IMGW_URL = "https://rafalraczynski.com.pl/imgw/dane-imgw/getJSON.php?type=table&province={prov}&sort=temp&order=asc"
IMGW_METEO_URL = "https://danepubliczne.imgw.pl/api/data/meteo/"
IMGW_HYDRO_URL = "https://danepubliczne.imgw.pl/api/data/hydro/"

def get_imgw_official_coords() -> dict:
    """
    Fetch official IMGW station coordinates from public API.

    Returns:
    - dict mapping station_name -> (lat, lon).
    """
    coords = {}
    
    # Fetch meteo stations
    try:
        response = requests.get(IMGW_METEO_URL, timeout=60)
        response.raise_for_status()
        for station in response.json():
            name = station.get('nazwa_stacji', '').strip().upper()
            lat, lon = station.get('lat'), station.get('lon')
            if name and lat and lon:
                coords[name] = (float(lat), float(lon))
        meteo_count = len(coords)
    except Exception as e:
        print(f"[IMGW] ⚠️ Meteo coords failed: {e}")
        meteo_count = 0
    
    # Fetch hydro stations
    try:
        response = requests.get(IMGW_HYDRO_URL, timeout=60)
        response.raise_for_status()
        for station in response.json():
            name = station.get('stacja', '').strip().upper()
            lat, lon = station.get('lat'), station.get('lon')
            if name and lat and lon and name not in coords:
                coords[name] = (float(lat), float(lon))
        hydro_count = len(coords) - meteo_count
    except Exception as e:
        print(f"[IMGW] ⚠️ Hydro coords failed: {e}")
        hydro_count = 0
    
    print(f"[IMGW] Loaded {len(coords)} official coordinates ({meteo_count} meteo + {hydro_count} hydro)")
    return coords

# Cache
_IMGW_COORDS_CACHE = None

def _get_cached_coords():
    """Get cached coordinates + fetching if needed"""
    global _IMGW_COORDS_CACHE
    if _IMGW_COORDS_CACHE is None:
        _IMGW_COORDS_CACHE = get_imgw_official_coords()
    return _IMGW_COORDS_CACHE

def fetch_imgw(provinces: List[int] = None) -> pd.DataFrame:
    """
    Fetch IMGW data for specified regions.
    
    Returns:
    - DataFrame with columns: station, temp, statId, source.
    """
    if provinces is None:
        provinces = IMGW_PROVINCES
    
    all_data = []

    session = requests.Session()
    
    retries = Retry(
        total=4,
        connect=3,
        read=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retries)

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        'Accept': 'application/json',
        'Accept-Language': 'pl-PL,pl;q=0.9,en-US;q=0.8'
    })

    for prov in provinces:
        for attempt in range(3):
            try:
                time.sleep(1 if attempt == 0 else 5)
                response = session.get(IMGW_URL.format(prov=prov), timeout=45)
                response.raise_for_status()
                data = response.json()
                all_data.extend(data)
                print(f"[IMGW] Province {prov:02d}: {len(data)} stations")
                break
            except Exception as e:
                if attempt < 2:
                    print(f"[IMGW] Province {prov:02d}: retry {attempt+1}/2 ({type(e).__name__})")
                else:
                    print(f"[IMGW] Province {prov:02d}: ❌ {e}")
    
    df = pd.DataFrame(all_data)
    if df.empty:
        return pd.DataFrame(columns=["station", "temp", "statId", "source"])
    
    # Filter by data mode (observations vs model)
    if 'isModel' in df.columns:
        obs_count = (~df['isModel']).sum()
        model_count = df['isModel'].sum()
        print(f"[IMGW] Data breakdown: {obs_count} observations, {model_count} model points")
        
        if IMGW_DATA_MODE == "observations":
            df = df[df['isModel'] == False]
            print(f"[IMGW] Filtered to observations only: {len(df)} stations")
        elif IMGW_DATA_MODE == "model":
            df = df[df['isModel'] == True]
            print(f"[IMGW] Filtered to model data only: {len(df)} stations")
        # "all" mode keeps every data point
    else:
        print("[IMGW] ⚠️ 'isModel' field not found in API response. Using all data.")
    
    # Keep isModel flag for dynamic lapse rate calculation
    columns_to_keep = ["statName", "temp", "statId"]
    if 'isModel' in df.columns:
        columns_to_keep.append('isModel')
    if 'provName' in df.columns:
        columns_to_keep.append('provName')
    
    df = df[columns_to_keep].rename(columns={"statName": "station"})
    df["temp"] = df["temp"].apply(clean_temperature)
    df = df.dropna(subset=["temp"])
    df["source"] = "IMGW"
    
    # Apply official coordinates from IMGW public API
    coord_lookup = _get_cached_coords()
    if coord_lookup:
        df['lat'] = None
        df['lon'] = None
        matched = 0
        for idx in df.index:
            name = df.at[idx, 'station'].strip().upper()
            if name in coord_lookup:
                df.at[idx, 'lat'] = coord_lookup[name][0]
                df.at[idx, 'lon'] = coord_lookup[name][1]
                matched += 1
        print(f"[IMGW] Matched {matched}/{len(df)} stations with official coordinates")
    
    print(f"[IMGW] Total: {len(df)} stations with valid data")
    return df

# Traxelektronik
TRAX_URL = "https://www.traxelektronik.pl/pogoda/zbiorcza.php?RejID={}"
TRAX_NUMERIC = re.compile(r"^-?\d+(?:[.,]\d+)?$")

def _fetch_trax_region(region_id: int) -> List[Tuple[str, float, int]]:
    """Fetch data from one Trax region"""
    try:
        response = requests.get(TRAX_URL.format(region_id), timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        rows = soup.find_all("tr")[2:]  # skip header rows
        data = []
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 2:
                continue
            
            name = cols[0].get_text(strip=True)
            temp_str = cols[1].get_text(strip=True).replace(",", ".")
            
            if TRAX_NUMERIC.match(temp_str):
                data.append((name, float(temp_str), region_id))
        
        print(f"[TRAX] Region {region_id:3d}: {len(data)} stations")
        return data
    
    except Exception as e:
        print(f"[TRAX] Region {region_id:3d}: ❌ {e}")
        return []

def fetch_trax(region_ids: List[int] = None) -> pd.DataFrame:
    """
    Fetch TraxElektronik data.
    
    Returns:
    - DataFrame with columns: station, temp, region, source.
    """
    if region_ids is None:
        region_ids = TRAX_REGION_IDS
    
    all_data = []
    for rid in region_ids:
        all_data.extend(_fetch_trax_region(rid))
    
    df = pd.DataFrame(all_data, columns=["station", "temp", "region"])
    df["source"] = "TRAX"
    
    print(f"[TRAX] Total: {len(df)} stations")
    return df

# Netatmo
def fetch_netatmo() -> pd.DataFrame:
    """
    Fetch Netatmo data.
    
    Returns:
    - DataFrame with columns: station, temp, lat, lon, source.
    """
    url = "https://api.netatmo.com/api/getpublicdata"
    params = {
        "lat_ne": NETATMO_CONFIG["lat_ne"],
        "lon_ne": NETATMO_CONFIG["lon_ne"],
        "lat_sw": NETATMO_CONFIG["lat_sw"],
        "lon_sw": NETATMO_CONFIG["lon_sw"],
        "required_data": "temperature",
        "filter": "true",
        "access_token": NETATMO_CONFIG["access_token"]
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"[NETATMO] ❌ {e}")
        return pd.DataFrame(columns=["station", "temp", "lat", "lon", "source"])
    
    records = []
    for station in data.get('body', []):
        place = station.get('place', {})
        location = place.get('location')
        
        if not location:
            continue
        
        lon, lat = location
        
        # check if in Poland
        if not is_in_poland(lat, lon):
            continue
        
        city = place.get('city', 'Unknown')
        
        # extract temperature
        for module_id, module_data in station.get('measures', {}).items():
            if 'temperature' not in module_data.get('type', []):
                continue
            
            res = module_data.get('res', {})
            if res:
                latest_timestamp = max(res.keys())
                temp = res[latest_timestamp][0]
                records.append({
                    "station": city,
                    "temp": temp,
                    "lat": lat,
                    "lon": lon,
                    "source": "NETATMO"
                })
                break
    
    df = pd.DataFrame(records)
    print(f"[NETATMO] Total: {len(df)} stations")
    return df

# Edwin
def _fetch_edwin_station_data(session, station_id: int, start_date: str) -> dict:
    """Fetch latest observation from one Edwin station."""
    try:
        url = f"{EDWIN_CONFIG['api_base']}/meteo/station/{station_id}?page=0&size=100&after={start_date}"
        res = session.get(url, timeout=10)
        if res.ok:
            content = res.json().get('content', [])
            if content:
                data = content[-1]  # most recent observation
                data.pop('links', None)
                return data
    except:
        pass
    return None

def fetch_edwin() -> pd.DataFrame:
    """
    Fetch Edwin data using concurrent requests.
    
    Returns:
    - DataFrame with columns: station, temp, lat, lon, source.
    """
    api_base = EDWIN_CONFIG['api_base']
    
    # Fetch station metadata
    meta_frames = []
    for stype in EDWIN_CONFIG['station_types']:
        try:
            url = f"{api_base}/observationStation?active=true&size=10000&sort=asc&type={stype}"
            res = requests.get(url, timeout=15)
            if res.ok:
                stations = res.json().get('content', [])
                meta_frames.append(pd.DataFrame(stations))
                print(f"[EDWIN] Loaded {len(stations)} {stype} stations")
        except Exception as e:
            print(f"[EDWIN] ⚠️ Metadata failed ({stype}): {e}")
    
    if not meta_frames:
        print("[EDWIN] ❌ No station metadata available")
        return pd.DataFrame(columns=["station", "temp", "lat", "lon", "source"])
    
    df_meta = pd.concat(meta_frames, ignore_index=True)
    
    # Time range for observations
    start_date = (datetime.now(timezone.utc) - timedelta(hours=EDWIN_CONFIG['lookback_hours'])).strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Concurrent fetching
    results = []
    with requests.Session() as session:
        with concurrent.futures.ThreadPoolExecutor(max_workers=EDWIN_CONFIG['workers']) as executor:
            futures = [executor.submit(_fetch_edwin_station_data, session, sid, start_date) for sid in df_meta['id']]
            for future in concurrent.futures.as_completed(futures):
                if data := future.result():
                    results.append(data)
    
    if not results:
        print("[EDWIN] ❌ No recent observations")
        return pd.DataFrame(columns=["station", "temp", "lat", "lon", "source"])
    
    df_obs = pd.DataFrame(results)
    
    # Merge observations with metadata to receive coordinates
    df = df_obs.merge(df_meta[['id', 'name', 'latitude', 'longitude']], 
                      left_on='stationId', right_on='id', how='left')
    
    # Extract temperature and build output DF
    records = []
    for _, row in df.iterrows():
        temp = row.get('airTemperature')
        lat = row.get('latitude')
        lon = row.get('longitude')
        name = row.get('name', 'Unknown')
        
        # Skip if no temperature/coordinates
        if pd.isna(temp) or pd.isna(lat) or pd.isna(lon):
            continue
        
        # Filter sensor failures
        humidity = row.get('relativeHumidity')
        if humidity == 0.0:
            continue
        
        records.append({
            "station": name,
            "temp": float(temp),
            "lat": float(lat),
            "lon": float(lon),
            "source": "EDWIN"
        })
    
    df_final = pd.DataFrame(records)
    print(f"[EDWIN] Total: {len(df_final)} stations with valid data")
    return df_final

# Combined fetching

import numpy as np
from scipy.spatial import cKDTree

def _deduplicate_stations(df, radius_m=100):
    """
    Remove duplicate stations across sources using spatial proximity.
    
    When two stations from different networks are within `radius_m` of each
    other, keep the one from the higher-priority source.
    
    Priority: IMGW > TRAX > EDWIN > NETATMO
    """
    if len(df) < 2:
        return df
    
    has_coords = df.dropna(subset=['lat', 'lon']).copy()
    no_coords = df[df['lat'].isna() | df['lon'].isna()]
    
    if len(has_coords) < 2:
        return df
    
    SOURCE_PRIORITY = {
        'IMGW': 0, 'TRAX': 1, 'EDWIN': 2,
        'NETATMO': 3,
    }
    
    lats = has_coords['lat'].values.astype(float)
    lons = has_coords['lon'].values.astype(float)
    
    # Approximate meter conversion at Poland's mean latitude (~52°N)
    cos52 = np.cos(np.radians(52.0))
    x = lons * 111_320 * cos52
    y = lats * 110_540
    
    tree = cKDTree(np.column_stack([x, y]))
    pairs = tree.query_pairs(radius_m)
    
    to_remove = set()
    for i, j in pairs:
        src_i = SOURCE_PRIORITY.get(has_coords.iloc[i]['source'], 99)
        src_j = SOURCE_PRIORITY.get(has_coords.iloc[j]['source'], 99)
        if src_i <= src_j:
            to_remove.add(has_coords.index[j])
        else:
            to_remove.add(has_coords.index[i])
    
    if to_remove:
        print(f"[DEDUP] Removed {len(to_remove)} duplicate stations "
              f"(within {radius_m}m across sources)")
        has_coords = has_coords.drop(index=to_remove)
    
    return pd.concat([has_coords, no_coords], ignore_index=True)

def fetch_all_data() -> pd.DataFrame:
    """
    Fetch data from all sources and combine.
    
    Sources (in priority order):
    - IMGW, Trax, Edwin, Netatmo.
    
    Returns:
    - DataFrame with columns: station, temp, lat, lon, source.
    - (lat/lon are None for IMGW/TRAX until geocoded).
    """
    print("Fetching data from all sources...")
    
    # Core sources
    imgw_df = fetch_imgw()
    trax_df = fetch_trax()
    netatmo_df = fetch_netatmo()
    edwin_df = fetch_edwin()
    
    # add lat/lon columns to IMGW and Traxelektronik (will be filled during geocoding)
    for df in [imgw_df, trax_df]:
        if not df.empty and 'lat' not in df.columns:
            df['lat'] = None
            df['lon'] = None
    
    all_dfs = [df for df in [imgw_df, trax_df, netatmo_df, edwin_df] if not df.empty]

    # Combine
    if not all_dfs:
        print("\u26a0\ufe0f  No data fetched from any source!")
        return pd.DataFrame(columns=["station", "temp", "lat", "lon", "source"])
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # ensure that columns are consistent, preserve isModel and provName if available
    core_cols = ["station", "temp", "lat", "lon", "source"]
    extra_cols = [c for c in ["isModel", "provName"] if c in combined.columns]
    combined = combined[core_cols + extra_cols]
    
    # Cross-source deduplication
    n_before = len(combined)
    combined = _deduplicate_stations(combined, radius_m=PWS_DEDUP_RADIUS_M)
    
    # Report isModel stats if available
    if 'isModel' in combined.columns:
        credible_count = (combined['isModel'] == False).sum()
        print(f"[IMGW] {credible_count} credible IMGW observations are available for dynamic lapse rate calculation.")
    
    print(f"Total stations fetched: {len(combined)} (before dedup: {n_before})")
    print(combined.groupby("source").size())

    return combined
