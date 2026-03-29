"""
Prepare station data. Geocoding and Spatial Quality Control (QC).
FS-ISCT: Feature-Space Iterative Spatial Consistency Test.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
from sklearn.linear_model import HuberRegressor
from tqdm import tqdm

from .config import (
    CRS_WGS84, CRS_POLAND,
    PERFORM_SPATIAL_QC,
    QC_SOURCE_PRIORS, QC_SOURCE_TOLERANCES,
    QC_HARD_REJECT_THRESHOLD, QC_CONFIDENCE_MIN, QC_BUTTERWORTH_ORDER,
    QC_ITERATIONS, QC_ANCHOR_WEIGHT,
    QC_DECLUSTER_RADIUS_KM,
    QC_KERNEL_GEO_KM, QC_KERNEL_DEM_M, QC_KERNEL_CAP, QC_KERNEL_SETTLEMENT, QC_KERNEL_SVF,
    QC_ISOLATION_ALPHA, QC_MAX_NEIGHBORS, QC_MAX_SEARCH_RADIUS_KM,
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
    
    # Check if provName column exists
    has_provname = 'provName' in df.columns
    
    for idx in tqdm(to_geocode.index, desc="Geocoding"):
        station_name = clean_station_name(df.loc[idx, 'station'])
        source = df.loc[idx, 'source']
        
        # Use provName for IMGW stations to improve geocoding accuracy
        prov_hint = None
        if has_provname and source == 'IMGW':
            prov_val = df.loc[idx, 'provName']
            if pd.notna(prov_val) and prov_val:
                prov_hint = str(prov_val).lower()
            
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
    Feature-Space Iterative Spatial Consistency Test (FS-ISCT).
    
    Instead of binary keep/reject, assigns each station a continuous confidence
    weight ∈ [C_min, 1.0] based on spatial consistency. Confidence flows into
    downstream LightGBM sample_weight and Kriging nugget scaling.
    """
    if not PERFORM_SPATIAL_QC or len(gdf) < 20 or 'dem' not in gdf.columns:
        print("\n⚠️  Skipping Spatial QC (not enabled, too few points, or no DEM).")
        gdf['qc_confidence'] = 1.0
        return gdf

    print(f"\nPerforming FS-ISCT Spatial Quality Control...")
    N = len(gdf)
    
    # Setup
    # Project to EPSG:2180 for meter-based distances
    gdf_proj = gdf.to_crs(CRS_POLAND)
    coords_m = np.array([[g.x, g.y] for g in gdf_proj.geometry])
    coords_km = coords_m / 1000.0
    
    temps = gdf['temp'].values.copy()
    sources = gdf['source'].values if 'source' in gdf.columns else np.full(N, 'UNKNOWN')
    
    # Source priors and tolerances
    P_i = np.array([QC_SOURCE_PRIORS.get(s, 0.4) for s in sources])
    tau_base = np.array([QC_SOURCE_TOLERANCES.get(s, 2.0) for s in sources])
    is_trusted = P_i >= 0.8  # IMGW and EDWIN
    
    n_trusted = is_trusted.sum()
    n_pws = (~is_trusted).sum()
    print(f"   Stations: {N} total ({n_trusted} trusted, {n_pws} PWS)")
    
    # Spatial Declustering
    # Prevent dense PWS clusters from outvoting sparse trusted networks
    R_dec = QC_DECLUSTER_RADIUS_KM  # km
    rho = np.ones(N)
    
    for src in np.unique(sources):
        src_mask = sources == src
        n_src = src_mask.sum()
        if n_src <= 1:
            continue
        
        src_indices = np.where(src_mask)[0]
        src_coords = coords_km[src_mask]
        tree_src = cKDTree(src_coords)
        
        # For each station of this source, count same-source neighbors
        # using Gaussian kernel within declustering radius
        pairs = tree_src.query_pairs(r=R_dec * 3.0, output_type='ndarray')
        
        if len(pairs) > 0:
            for local_i in range(n_src):
                # Find neighbors of this station within radius
                nbr_dists = np.linalg.norm(src_coords[local_i] - src_coords, axis=1)
                within = nbr_dists < R_dec * 3.0
                rho_val = np.sum(np.exp(-(nbr_dists[within] / R_dec) ** 2))
                rho[src_indices[local_i]] = max(1.0, rho_val)
    
    w_base = P_i / rho
    
    decluster_stats = {}
    for src in np.unique(sources):
        mask = sources == src
        avg_rho = rho[mask].mean()
        if avg_rho > 1.1:
            decluster_stats[src] = f"ρ={avg_rho:.1f}"
    if decluster_stats:
        print(f"   Declustering: {decluster_stats}")
    
    # Microclimate-Adjusted Background
    # Fit robust regression on trusted stations to learn terrain→temperature
    dems = gdf['dem'].fillna(0).values
    
    # Gather feature columns with safe fallbacks
    feat_names = []
    feat_arrays = []
    
    for col in ['dem', 'cap_anomaly', 'settlement', 'tpi_500']:
        if col in gdf.columns:
            vals = gdf[col].fillna(0).values.astype(float)
            feat_names.append(col)
            feat_arrays.append(vals)
    
    if len(feat_arrays) == 0:
        # Bare minimum: just elevation
        feat_names = ['dem']
        feat_arrays = [dems]
    
    X_raw = np.column_stack(feat_arrays)
    
    # Standardize features for Huber regression stability
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    huber = HuberRegressor(epsilon=1.35, max_iter=300)
    
    # Decide NWP vs spatial-trend background
    nwp_available = ('nwp_temp' in gdf.columns and 
                     gdf['nwp_temp'].notna().sum() > max(30, N * 0.3))
    
    if nwp_available:
        nwp_raw = gdf['nwp_temp'].values.copy()
        nwp_temps = np.where(np.isnan(nwp_raw), temps, nwp_raw)  # fallback to obs where NWP missing
        target = temps - nwp_temps  # innovation: obs - NWP
        
        # Fit on trusted stations only
        trusted_count = is_trusted.sum()
        if trusted_count >= 10:
            huber.fit(X_scaled[is_trusted], target[is_trusted])
            B = nwp_temps + huber.predict(X_scaled)
            print(f"   MAB: NWP-anchored (fitted on {trusted_count} trusted stations)")
        else:
            B = nwp_temps
            print(f"   MAB: NWP raw (too few trusted: {trusted_count})")
    else:
        # No NWP: build spatial-trend surface from coordinates + terrain
        X_macro = np.column_stack([coords_km, coords_km ** 2, X_scaled])
        trusted_count = is_trusted.sum()
        
        if trusted_count >= 10:
            huber.fit(X_macro[is_trusted], temps[is_trusted])
            B = huber.predict(X_macro)
            print(f"   MAB: Spatial-trend surface (fitted on {trusted_count} trusted stations)")
        else:
            # Fallback: use dynamic lapse rate if available
            if lapse_rate is not None:
                B = np.median(temps) - (dems - np.median(dems)) * lapse_rate
                print(f"   MAB: Lapse-rate fallback ({lapse_rate:.4f} °C/m)")
            else:
                B = np.full(N, np.median(temps))
                print(f"   MAB: Median fallback ({np.median(temps):.1f}°C)")
    
    # Innovation: departure from background
    v = temps - B
    mab_rmse = np.sqrt(np.mean(v[is_trusted] ** 2)) if is_trusted.sum() > 0 else np.nan
    print(f"   MAB residual RMSE (trusted): {mab_rmse:.2f}°C")
    
    # Feature-Space Neighbor Search
    # Find geographic neighbors, then weight by feature-space similarity
    tree_all = cKDTree(coords_km)
    max_r_km = QC_MAX_SEARCH_RADIUS_KM
    k_max = min(QC_MAX_NEIGHBORS, N - 1)
    
    dists_km, nn_idx = tree_all.query(coords_km, k=k_max + 1, 
                                        distance_upper_bound=max_r_km)
    # Remove self (column 0) 
    dists_km = dists_km[:, 1:]
    nn_idx = nn_idx[:, 1:]
    
    # Pad arrays for out-of-bounds indexing (cKDTree returns N for OOB)
    v_pad = np.append(v, 0.0)
    w_pad = np.append(w_base, 0.0)
    
    # Feature arrays for kernel computation
    dem_arr = dems.copy()
    cap_arr = gdf['cap_anomaly'].fillna(0).values if 'cap_anomaly' in gdf.columns else np.zeros(N)
    set_arr = gdf['settlement'].fillna(0).values if 'settlement' in gdf.columns else np.zeros(N)
    svf_arr = gdf['svf'].fillna(1.0).values if 'svf' in gdf.columns else np.ones(N)
    
    dem_pad = np.append(dem_arr, 0.0)
    cap_pad = np.append(cap_arr, 0.0)
    set_pad = np.append(set_arr, 0.0)
    svf_pad = np.append(svf_arr, 1.0)
    
    # 5D Feature-space distance²
    L_geo = QC_KERNEL_GEO_KM
    L_dem = QC_KERNEL_DEM_M / 1000.0  # convert to km-scale for dists_km comparison
    L_cap = QC_KERNEL_CAP
    L_set = QC_KERNEL_SETTLEMENT
    L_svf = QC_KERNEL_SVF
    
    D2 = (dists_km / L_geo) ** 2
    D2 += ((dem_arr[:, None] - dem_pad[nn_idx]) / QC_KERNEL_DEM_M) ** 2
    D2 += ((cap_arr[:, None] - cap_pad[nn_idx]) / L_cap) ** 2
    D2 += ((set_arr[:, None] - set_pad[nn_idx]) / L_set) ** 2
    D2 += ((svf_arr[:, None] - svf_pad[nn_idx]) / L_svf) ** 2
    
    K_ij = np.exp(-0.5 * D2)
    
    # Zero out self-matches and out-of-bounds
    K_ij[nn_idx == N] = 0.0
    # Self-exclusion (if somehow self appears in neighbors beyond col 0)
    self_mask = nn_idx == np.arange(N)[:, None]
    K_ij[self_mask] = 0.0
    
    # Iterative Bayesian Expected Residual
    W_anchor = QC_ANCHOR_WEIGHT
    alpha_iso = QC_ISOLATION_ALPHA
    C_min = QC_CONFIDENCE_MIN
    bw_order = QC_BUTTERWORTH_ORDER
    n_iter = QC_ITERATIONS
    
    C = np.ones(N)
    
    for iteration in range(n_iter):
        C_pad = np.append(C, 0.0)
        
        W_ij = C_pad[nn_idx] * w_pad[nn_idx] * K_ij
        
        S_i = np.sum(W_ij, axis=1)
        
        v_hat = np.sum(W_ij * v_pad[nn_idx], axis=1) / (S_i + W_anchor)
        
        e_i = np.abs(v - v_hat)
        
        tau_i = tau_base + alpha_iso * np.exp(-S_i / W_anchor)
        
        # Butterworth confidence filter
        C = C_min + (1.0 - C_min) / (1.0 + (e_i / tau_i) ** bw_order)
    
    # Hard rejection of physical impossibilities
    hard_reject = e_i > QC_HARD_REJECT_THRESHOLD
    n_hard_reject = hard_reject.sum()
    
    # Results
    gdf = gdf.copy()
    gdf['qc_confidence'] = C
    
    # Stats
    high_conf = (C > 0.8).sum()
    med_conf = ((C > 0.3) & (C <= 0.8)).sum()
    low_conf = ((C > 0.05) & (C <= 0.3)).sum()
    very_low = (C <= 0.05).sum()
    
    print(f"\n   FS-ISCT Results ({n_iter} iterations):")
    print(f"      High confidence (>0.8):  {high_conf:4d} stations")
    print(f"      Medium (0.3–0.8):        {med_conf:4d} stations")
    print(f"      Low (0.05–0.3):          {low_conf:4d} stations")
    print(f"      Very low (≤0.05):        {very_low:4d} stations")
    
    # Per-source breakdown
    source_summary = {}
    for src in np.unique(sources):
        mask = sources == src
        src_C = C[mask]
        source_summary[src] = f"μ={src_C.mean():.2f}"
    print(f"      By source: {source_summary}")
    
    # Hard rejections
    if n_hard_reject > 0:
        reject_idx = np.where(hard_reject)[0]
        print(f"      ❌ Hard rejected ({QC_HARD_REJECT_THRESHOLD}°C+): {n_hard_reject}")
        for idx in reject_idx[:5]:
            row = gdf.iloc[idx]
            print(f"         - {row['station']} ({row['source']}): "
                  f"T={row['temp']:.1f}°C, Dev={e_i[idx]:.1f}°C")
        gdf = gdf[~hard_reject].copy()
    
    print(f"   ✓ QC Complete. {len(gdf)} stations with confidence weights.")
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
    
    # Deduplicate locations (prioritize by coordinate accuracy)
    source_priority = {'IMGW': 1, 'EDWIN': 2, 'NETATMO': 3, 'TRAX': 4}
    gdf['priority'] = gdf['source'].map(source_priority).fillna(99)
    gdf = gdf.sort_values('priority')
    
    # Keep first (highest priority) at unique lat/lon
    before = len(gdf)
    gdf = gdf.drop_duplicates(subset=['lat', 'lon'], keep='first')
    gdf = gdf.drop(columns=['priority'])
    
    print(f"\n✓ Prepared {len(gdf)} unique stations (dropped {before-len(gdf)} duplicates)")
        
    return gdf