"""
Fetch temperature data from IMGW, Netatmo, and TraxElektronik.
"""
import re
import json
from typing import List, Tuple
import requests
import pandas as pd
from bs4 import BeautifulSoup

from .config import IMGW_PROVINCES, IMGW_DATA_MODE, TRAX_REGION_IDS, NETATMO_CONFIG
from .utils import is_in_poland, clean_temperature

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# IMGW
IMGW_URL = "https://rafalraczynski.com.pl/imgw/dane-imgw/getJSON.php?type=table&province={prov}&sort=temp&order=asc"

def get_session():
    """Create a requests session with retries and timeout."""
    session = requests.Session()
    retry = Retry(
        total=5, 
        backoff_factor=1, 
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    })
    return session

def fetch_imgw(provinces: List[int] = None) -> pd.DataFrame:
    """
    Fetch IMGW data for specified regions
    
    Returns:
        DataFrame with columns: station, temp, statId, source
    """
    if provinces is None:
        provinces = IMGW_PROVINCES
    
    all_data = []
    session = get_session()
    
    for prov in provinces:
        try:
            time.sleep(0.1)
            response = session.get(IMGW_URL.format(prov=prov), timeout=30)
            response.raise_for_status()
            data = response.json()
            all_data.extend(data)
            print(f"[IMGW] Province {prov:02d}: {len(data)} stations")
        except Exception as e:
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
    
    df = df[["statName", "temp", "statId"]].rename(columns={"statName": "station"})
    df["temp"] = df["temp"].apply(clean_temperature)
    df = df.dropna(subset=["temp"])
    df["source"] = "IMGW"
    
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
    Fetch TraxElektronik data
    
    Returns:
        DataFrame with columns: station, temp, region, source
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
    Fetch Netatmo data
    
    Returns:
        DataFrame with columns: station, temp, lat, lon, source
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

# Combined fetching
def fetch_all_data() -> pd.DataFrame:
    """
    Fetch data from all sources and combine
    
    Returns:
        DataFrame with columns: station, temp, lat, lon, source
        (lat/lon are None for IMGW/TRAX until geocoded)
    """
    print("Fetching data from all sources...")
    
    # Fetch from each source
    imgw_df = fetch_imgw()
    trax_df = fetch_trax()
    netatmo_df = fetch_netatmo()
    
    # add lat/lon columns to IMGW and Traxelektronik (will be filled during geocoding)
    for df in [imgw_df, trax_df]:
        if not df.empty and 'lat' not in df.columns:
            df['lat'] = None
            df['lon'] = None
    
    # combine all
    all_dfs = [df for df in [imgw_df, trax_df, netatmo_df] if not df.empty]
    
    if not all_dfs:
        print("⚠️  No data fetched from any source!")
        return pd.DataFrame(columns=["station", "temp", "lat", "lon", "source"])
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # ensure that columns are consistent
    combined = combined[["station", "temp", "lat", "lon", "source"]]
    
    print(f"Total stations fetched: {len(combined)}")
    print(combined.groupby("source").size())

    return combined
