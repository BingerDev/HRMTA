"""
Clean and professional visualization pipeline for HRMTA.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.colors import LinearSegmentedColormap, Normalize, ListedColormap, BoundaryNorm
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.image as mpimg
from pathlib import Path
import matplotlib.transforms as mtransforms
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from math import cos, radians
from datetime import datetime

from .config import (
    COLOR_SCALE, DPI, OUTPUT_PLOT,
    GRID_RESOLUTION, DISPLAY_STATION_SOURCES, DISPLAY_OBSERVATIONS_ONLY,
    INTERPOLATION_REGION, CRS_WGS84, DISPLAY_COUNTIES,
    COUNTIES_SHAPEFILE, VOIVODESHIP_SHAPEFILE,
    DISPLAY_CONTOURS,
    CONTOUR_INTERVAL, DISPLAY_CLEAN_MODE
)
from .utils import PL_BOUNDARY_WGS84, load_region_boundary, VOIVODESHIP_NAMES, is_national_mode

def load_color_scale_advanced(data_range=None):
    """
    Load color scale from CSV control points with sub-degree resolution.
    """
    _SUBS_PER_BAND = 2
    _INTRA_BLEND = 0.4

    if not COLOR_SCALE.exists():
        print(f"⚠️  Color scale not found: {COLOR_SCALE}")
        cmap = plt.cm.RdYlBu_r
        norm = Normalize(vmin=-40, vmax=40)
        return cmap, norm, -40, 40

    try:
        df = pd.read_csv(COLOR_SCALE).sort_values("value")

        if 'value' not in df.columns or 'color' not in df.columns:
            print("⚠️  Invalid color scale format")
            return plt.cm.RdYlBu_r, Normalize(vmin=-40, vmax=40), -40, 40

        temps = df['value'].to_numpy(dtype=float)
        colors_rgb = [np.array(mcolors.to_rgb(c)) for c in df['color'].values]

        # Clip to data range with margin
        if data_range is not None:
            margin = 3.0
            clip_lo = max(temps[0], np.floor(data_range[0] - margin))
            clip_hi = min(temps[-1], np.ceil(data_range[1] + margin))
        else:
            clip_lo, clip_hi = temps[0], temps[-1]

        # Build sub-bin colors
        all_bounds = []
        all_colors = []

        for i in range(len(temps) - 1):
            t_lo, t_hi = temps[i], temps[i + 1]
            c_lo, c_hi = colors_rgb[i], colors_rgb[i + 1]
            band_w = t_hi - t_lo

            if t_hi <= clip_lo or t_lo >= clip_hi:
                continue

            for s in range(_SUBS_PER_BAND):
                sub_t = t_lo + s * band_w / _SUBS_PER_BAND
                if sub_t < clip_lo or sub_t >= clip_hi:
                    continue

                # Blend toward next band, limited by _INTRA_BLEND
                frac = (s / _SUBS_PER_BAND) * _INTRA_BLEND
                color = tuple(c_lo + frac * (c_hi - c_lo))

                all_bounds.append(sub_t)
                all_colors.append(color)

        all_bounds.append(clip_hi)

        if len(all_colors) < 2:
            return plt.cm.RdYlBu_r, Normalize(vmin=clip_lo, vmax=clip_hi), clip_lo, clip_hi

        bounds = np.array(all_bounds)
        cmap_fine = ListedColormap(all_colors, name="hrmta_fine")
        norm = BoundaryNorm(bounds, cmap_fine.N, clip=True)

        bin_w = np.median(np.diff(bounds))
        print(f"✓ Color scale: {clip_lo:.0f}°C to {clip_hi:.0f}°C "
              f"({len(all_colors)} bins, ~{bin_w:.1f}°C step)")
        return cmap_fine, norm, clip_lo, clip_hi

    except Exception as e:
        print(f"⚠️  Error loading color scale: {e}")
        return plt.cm.RdYlBu_r, Normalize(vmin=-40, vmax=40), -40, 40

# Station marker sizes by source accuracy tier
_SOURCE_MARKER_SIZES = {
    'IMGW': 22, 'EDWIN': 16, 'TRAX': 12,
    'NETATMO': 5,
}
_DEFAULT_MARKER_SIZE = 8

def plot_temperature_map(
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    temperature: np.ndarray,
    stations_gdf: gpd.GeoDataFrame = None,
    output_path: str = None,
    show: bool = True,
    title_suffix: str = "",
    resolution_km=GRID_RESOLUTION,
    region_name: str = "Polska",
    data_fetch_time: datetime = None
):
    """
    Exact visualization style match:
    - Cosine-corrected aspect ratio
    - Discrete color steps
    - Specific station markers (black squares) and offsets
    - Min/Max callouts in bottom-left
    """
    if output_path is None:
        output_path = OUTPUT_PLOT

    # Load data & scale
    cmap, norm, scale_vmin, scale_vmax = load_color_scale_advanced()
    
    # Setup Figure Geometry based on region boundary
    region_gdf = load_region_boundary(INTERPOLATION_REGION, CRS_WGS84)
    region_bounds = region_gdf.total_bounds # [minx, miny, maxx, maxy]
    minx, miny, maxx, maxy = region_bounds
    
    # Add small margin
    margin_x = (maxx - minx) * 0.02
    margin_y = (maxy - miny) * 0.02
    minx -= margin_x
    maxx += margin_x
    miny -= margin_y
    maxy += margin_y
    
    lat_mid = (miny + maxy) / 2
    aspect_true = (maxx - minx) * cos(radians(lat_mid)) / (maxy - miny)
    
    FIG_W = 13  # Fixed width from reference
    FIG_H = FIG_W / aspect_true
    
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    
    # Plot Raster
    img = ax.pcolormesh(
        grid_lon, grid_lat, temperature,
        cmap=cmap,
        norm=norm,
        shading='nearest',
        rasterized=True,
        zorder=1
    )

    # Isotherm contours
    if DISPLAY_CONTOURS:
        temp_masked = np.ma.masked_invalid(temperature)
        data_min = float(np.nanmin(temperature))
        data_max = float(np.nanmax(temperature))
        contour_levels = np.arange(
            np.floor(data_min),
            np.ceil(data_max) + 0.1,
            CONTOUR_INTERVAL
        )
        ax.contour(
            grid_lon, grid_lat, temp_masked,
            levels=contour_levels,
            colors='#222222', linewidths=0.25, alpha=0.35,
            zorder=1.5
        )

    # Borders & Neatline
    if is_national_mode(INTERPOLATION_REGION):
        # National: country outline (bold) + voivodeships (medium) + counties (fine)
        region_gdf.boundary.plot(
            ax=ax, edgecolor="#222222", linewidth=1.2, zorder=2.2
        )
        if VOIVODESHIP_SHAPEFILE.exists():
            voi_gdf = gpd.read_file(VOIVODESHIP_SHAPEFILE).to_crs(CRS_WGS84)
            voi_gdf.boundary.plot(
                ax=ax, edgecolor="#444444", linewidth=0.5, zorder=2.1
            )
        if DISPLAY_COUNTIES and COUNTIES_SHAPEFILE.exists():
            counties_gdf = gpd.read_file(COUNTIES_SHAPEFILE).to_crs(CRS_WGS84)
            counties_gdf.boundary.plot(
                ax=ax, edgecolor="#777777", linewidth=0.12, alpha=0.5, zorder=2
            )
    else:
        # Regional: region outline (medium) + counties (fine)
        region_gdf.boundary.plot(
            ax=ax, edgecolor="#333333", linewidth=1.0, zorder=2.1
        )
        if DISPLAY_COUNTIES and COUNTIES_SHAPEFILE.exists():
            counties_gdf = gpd.read_file(COUNTIES_SHAPEFILE)
            canonical_name = VOIVODESHIP_NAMES.get(
                INTERPOLATION_REGION.lower(), INTERPOLATION_REGION.lower()
            )
            counties_in_region = counties_gdf[
                counties_gdf['NAME_1'].str.lower() == canonical_name
            ]
            if not counties_in_region.empty:
                counties_in_region = counties_in_region.to_crs(CRS_WGS84)
                counties_in_region.boundary.plot(
                    ax=ax, edgecolor="#888888", linewidth=0.25, zorder=2
                )
    
    # Box around extent
    ax.add_patch(Rectangle(
        (minx, miny),
        maxx - minx, maxy - miny,
        linewidth=1.0, edgecolor="#333333",
        facecolor="none", zorder=3
    ))
    
    # Plot Stations
    if stations_gdf is not None and not stations_gdf.empty:
        # Filter stations by source
        if 'source' in stations_gdf.columns and DISPLAY_STATION_SOURCES:
            stations_to_plot = stations_gdf[stations_gdf['source'].isin(DISPLAY_STATION_SOURCES)].copy()
        else:
            stations_to_plot = stations_gdf.copy()

        # Filter out IMGW model points if configured
        if DISPLAY_OBSERVATIONS_ONLY and 'isModel' in stations_to_plot.columns:
            stations_to_plot = stations_to_plot[stations_to_plot['isModel'] != True].copy()

        # Clip to map extent
        stations_to_plot = stations_to_plot.cx[minx:maxx, miny:maxy].copy()

        # Only proceed if we have stations left to plot
        if not stations_to_plot.empty:
            # assume temperature column exists
            temp_col = 'temp' if 'temp' in stations_to_plot.columns else 'temperature'
            if temp_col not in stations_to_plot.columns:
                # fallback for display if column not found
                stations_to_plot['temp_display'] = 0.0
                temp_col = 'temp_display'

        # Verify station sources being plotted
        if 'source' in stations_to_plot.columns:
            unique_sources = stations_to_plot['source'].unique()
            print(f"✓ Plotting {len(stations_to_plot)} stations from sources: {list(unique_sources)}")
        else:
            print(f"⚠ No 'source' column found. Plotting {len(stations_to_plot)} stations (source unverified)")

        # Markers (size by source accuracy)
        if 'source' in stations_to_plot.columns:
            marker_sizes = stations_to_plot['source'].map(
                _SOURCE_MARKER_SIZES
            ).fillna(_DEFAULT_MARKER_SIZE).values
            # IMGW model points get smaller markers (2-3°C error vs gold-standard obs)
            if 'isModel' in stations_to_plot.columns:
                is_model = stations_to_plot['isModel'].fillna(False).values
                marker_sizes = np.where(is_model, 8, marker_sizes)
        else:
            marker_sizes = 20

        ax.scatter(
            stations_to_plot.geometry.x,
            stations_to_plot.geometry.y,
            s=marker_sizes, marker="s", color="black",
            linewidths=0, zorder=4
        )
        
        # Labels
        txt_kw = dict(
            fontsize=6, color="black", weight="normal",
            ha="left", va="center", zorder=5,
            path_effects=[
                patheffects.Stroke(linewidth=0.7, foreground="white", alpha=0.8),
                patheffects.Normal()
            ]
        )
        
        dx_pt = 2.5  # shift 2.5 points right
        
        # Scale label grid with map extent
        map_width = maxx - minx
        label_grid_size = max(0.08, map_width / 40)
        occupied_cells = set()
        
        for idx, row in stations_to_plot.iterrows():
            if pd.isna(row[temp_col]): continue
            
            cell_x = int(row.geometry.x / label_grid_size)
            cell_y = int(row.geometry.y / label_grid_size)
            cell_key = (cell_x, cell_y)
            
            if cell_key in occupied_cells:
                continue
            occupied_cells.add(cell_key)
            
            label_text = f"{row[temp_col]:.1f}"
            txt = ax.text(row.geometry.x, row.geometry.y, label_text, **txt_kw)
            
            # Apply offset transform
            txt.set_transform(txt.get_transform() + 
                              mtransforms.ScaledTranslation(dx_pt/72, 0, fig.dpi_scale_trans))

    # Tmax/Tmin annotation
    if not DISPLAY_CLEAN_MODE and stations_gdf is not None and 'source' in stations_gdf.columns and 'temp' in stations_gdf.columns:
        # Filter for IMGW observational stations
        imgw_mask = stations_gdf['source'] == 'IMGW'
        if 'isModel' in stations_gdf.columns:
            imgw_mask = imgw_mask & (stations_gdf['isModel'] == False)
        
        imgw_obs = stations_gdf[imgw_mask]
        
        if not imgw_obs.empty:
            tmax_val = imgw_obs['temp'].max()
            tmin_val = imgw_obs['temp'].min()
            
            # Find ALL stations with max/min temperature (handles ties)
            tmax_stations = imgw_obs[imgw_obs['temp'] == tmax_val]
            tmin_stations = imgw_obs[imgw_obs['temp'] == tmin_val]
            
            # Build station names string
            tmax_names = ', '.join(tmax_stations['station'].str.title().tolist()) if 'station' in tmax_stations.columns else ''
            tmin_names = ', '.join(tmin_stations['station'].str.title().tolist()) if 'station' in tmin_stations.columns else ''
            
            # Build voivodeship strings (handle multiple)
            if 'provName' in imgw_obs.columns:
                tmax_provs = tmax_stations['provName'].dropna().unique()
                tmin_provs = tmin_stations['provName'].dropna().unique()
                tmax_woj = ', '.join([f"(woj. {p.lower()})" for p in tmax_provs]) if len(tmax_provs) > 0 else ''
                tmin_woj = ', '.join([f"(woj. {p.lower()})" for p in tmin_provs]) if len(tmin_provs) > 0 else ''
            else:
                tmax_woj = ''
                tmin_woj = ''
            
            text_x = minx + (maxx - minx) * 0.02
            text_y_tmax = miny + (maxy - miny) * 0.12
            text_y_tmin = miny + (maxy - miny) * 0.045
            name_offset = (maxy - miny) * 0.004
            woj_offset = (maxy - miny) * 0.003
            
            extrema_kw = dict(
                ha='left', zorder=10,
                path_effects=[
                    patheffects.Stroke(linewidth=3, foreground='white', alpha=0.9),
                    patheffects.Normal()
                ]
            )
            
            # Tmax
            ax.text(text_x, text_y_tmax, f"{tmax_val:.1f}°C",
                    fontsize=18, color='#FF0000', weight='bold', va='bottom', **extrema_kw)
            ax.text(text_x, text_y_tmax - name_offset, tmax_names,
                    fontsize=11, color='#FF0000', weight='bold', va='top', **extrema_kw)
            ax.text(text_x, text_y_tmax - name_offset - woj_offset - (maxy - miny) * 0.012, tmax_woj,
                    fontsize=7, color='#909090', weight='bold', va='top', **extrema_kw)
            
            # Tmin
            ax.text(text_x, text_y_tmin, f"{tmin_val:.1f}°C",
                    fontsize=18, color='#1A00FF', weight='bold', va='bottom', **extrema_kw)
            ax.text(text_x, text_y_tmin - name_offset, tmin_names,
                    fontsize=11, color='#1A00FF', weight='bold', va='top', **extrema_kw)
            ax.text(text_x, text_y_tmin - name_offset - woj_offset - (maxy - miny) * 0.012, tmin_woj,
                    fontsize=7, color='#909090', weight='bold', va='top', **extrema_kw)

    # Titles & Footer
    if data_fetch_time is not None:
        utc_now = data_fetch_time.strftime('%Y-%m-%d  %H:%M  UTC')
    else:
        from datetime import timezone
        utc_now = datetime.now(timezone.utc).strftime('%Y-%m-%d  %H:%M  UTC')
    
    if not DISPLAY_CLEAN_MODE:
        # Title aligned left
        if resolution_km >= 1000:
            res_str = f"{resolution_km / 1000:g} km"
        else:
            res_str = f"{resolution_km:g} m"
        ax.set_title(
            f'{region_name} • Temperatura powietrza 2 m • {res_str}\n{utc_now}',
            loc='left', pad=10, fontsize=12, weight='bold'
        )

        # Source footer
        ax.text(
            0.0, -0.01,
            'Źródła pochodzenia danych obserwacyjnych: IMGW (Instytut Meteorologii i Gospodarki Wodnej)  •  TraxElektronik  •  Netatmo  •  Edwin',
            transform=ax.transAxes,
            fontsize=6, color='#444444',
            ha='left', va='top'
        )
    
    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.12)
    
    cb = fig.colorbar(img, cax=cax)
    
    # Adaptive tick interval based on color scale range
    scale_span = scale_vmax - scale_vmin
    if scale_span <= 20:
        tick_step = 1
    elif scale_span <= 40:
        tick_step = 2
    else:
        tick_step = 5
    
    ticks = np.arange(
        np.ceil(scale_vmin / tick_step) * tick_step,
        np.floor(scale_vmax / tick_step) * tick_step + 0.5,
        tick_step, dtype=int
    )
    # Below -20°C show only -20 and -40 (skip intermediate ticks)
    ticks = ticks[(ticks >= -20) | (ticks == -40)]
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticks)
    cb.ax.tick_params(labelsize=9)
    cb.set_label('Temperatura 2 m (°C)', rotation=270, labelpad=15, fontsize=9)
    
    # Final settings
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect('auto') # handled aspect in figsize
    ax.axis('off')

    # Adaptive DPI: ensure grid pixels are resolved at high resolutions
    grid_cols = temperature.shape[1] if temperature.ndim == 2 else 1000
    data_width_in = FIG_W * 0.82
    output_dpi = max(DPI, min(int(np.ceil(grid_cols / data_width_in)), 600))

    plt.tight_layout()
    plt.savefig(output_path, dpi=output_dpi, bbox_inches='tight')
    print(f"\n✓ Map saved to: {output_path} (DPI={output_dpi})")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_uncertainty_map(
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    uncertainty: np.ndarray,
    output_path: Path,
    title: str = "Prediction Uncertainty"
):
    """Uncertainty map."""
    region_gdf = load_region_boundary(INTERPOLATION_REGION, CRS_WGS84)
    region_bounds = region_gdf.total_bounds
    minx, miny, maxx, maxy = region_bounds
    
    margin_x = (maxx - minx) * 0.02
    margin_y = (maxy - miny) * 0.02
    minx -= margin_x
    maxx += margin_x
    miny -= margin_y
    maxy += margin_y
    
    center_lat = (miny + maxy) / 2
    aspect_true = (maxx - minx) * cos(radians(center_lat)) / (maxy - miny)
    
    FIG_W = 10
    FIG_H = FIG_W / aspect_true
    
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor('#f8f9fa')
    
    # uncertainty colormap
    unc_cmap = LinearSegmentedColormap.from_list(
        'uncertainty',
        ['#ffffff', '#fff5f5', '#ffe5e5', '#ffc9c9', '#ff8787', '#ff6b6b', '#fa5252', '#f03e3e']
    )
    
    # plot uncertainty
    img = ax.pcolormesh(
        grid_lon, grid_lat, uncertainty,
        cmap=unc_cmap,
        shading='auto',
        vmin=0,
        vmax=np.nanpercentile(uncertainty, 98),  # clip to 98th percentile
        rasterized=True
    )
    
    # Region boundary
    region_gdf.boundary.plot(ax=ax, edgecolor='#2c3e50', linewidth=1.5)
    
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect('auto')
    ax.axis('off')
    
    # Title
    subtitle = f"Mean: {np.nanmean(uncertainty):.3f}°C, 95th percentile: {np.nanpercentile(uncertainty, 95):.3f}°C"
    ax.set_title(f"{title}\n{subtitle}", fontsize=14, fontweight='bold', 
                 color='#2c3e50', pad=10, loc='left')
    
    # Colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.12)
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label('Uncertainty (°C)', fontsize=12, color='#2c3e50', weight='bold')
    cbar.ax.tick_params(labelsize=10, colors='#2c3e50')
    cbar.outline.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='#f8f9fa')
    print(f"✓ Uncertainty map saved to: {output_path}")

def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20):
    """Modern feature importance plot."""
    from .config import RUN_OUTPUT_DIR
    
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#f8f9fa')
    
    y_pos = np.arange(len(top_features))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    
    bars = ax.barh(y_pos, top_features['importance'], color=colors, 
                   edgecolor='white', linewidth=1.5)
    
    if 'importance_std' in top_features.columns:
        ax.errorbar(top_features['importance'], y_pos, 
                   xerr=top_features['importance_std'],
                   fmt='none', ecolor='#2c3e50', capsize=3, linewidth=1.5, alpha=0.6)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'], fontsize=11)
    ax.set_xlabel('Importance', fontsize=13, weight='bold', color='#2c3e50')
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=16, 
                weight='bold', color='#2c3e50', pad=20)
    ax.invert_yaxis()
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#dee2e6')
    ax.spines['bottom'].set_color('#dee2e6')
    
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(RUN_OUTPUT_DIR / 'feature_importance.png', dpi=200, 
                bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()
    
    print("✓ Feature importance plot saved")

def plot_model_comparison(
    comparison_df: pd.DataFrame,
    test_results_baseline: pd.DataFrame,
    test_results_hybrid: pd.DataFrame,
    output_path: Path = None
):
    """
    Create comprehensive model comparison visualization
    
    Shows:
    1. Metrics comparison bar chart
    2. Scatter plot: predicted vs actual (both models)
    3. Error distribution histograms
    """
    if output_path is None:
        from .config import RUN_OUTPUT_DIR
        output_path = RUN_OUTPUT_DIR / "model_comparison.png"
    
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('#f8f9fa')
    
    # Title
    fig.suptitle('Model Comparison: Simple Kriging vs HRMTA', 
                 fontsize=24, fontweight='bold', color='#2c3e50', y=0.98)
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3,
                         left=0.08, right=0.95, top=0.92, bottom=0.08)
    
    # Metrics comparison
    ax1 = fig.add_subplot(gs[0, :2])
    
    metrics_to_plot = ['RMSE', 'MAE', 'R²']
    x_pos = np.arange(len(metrics_to_plot))
    width = 0.35
    
    baseline_vals = [comparison_df.loc[m, comparison_df.columns[0]] for m in metrics_to_plot]
    hybrid_vals = [comparison_df.loc[m, comparison_df.columns[1]] for m in metrics_to_plot]
    
    bars1 = ax1.bar(x_pos - width/2, baseline_vals, width, 
                    label='Simple Kriging', color='#95a5a6', edgecolor='white', linewidth=2)
    bars2 = ax1.bar(x_pos + width/2, hybrid_vals, width,
                    label='HRMTA', color='#3498db', edgecolor='white', linewidth=2)
    
    ax1.set_ylabel('Value', fontsize=13, fontweight='bold')
    ax1.set_title('Performance Metrics Comparison', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(metrics_to_plot, fontsize=12)
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Remove spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Improvement percentages
    ax2 = fig.add_subplot(gs[0, 2])
    
    improvements = [comparison_df.loc[m, '% Improvement'] for m in metrics_to_plot]
    colors = ['#27ae60' if imp > 0 else '#e74c3c' for imp in improvements]
    
    bars = ax2.barh(metrics_to_plot, improvements, color=colors, 
                    edgecolor='white', linewidth=2)
    
    ax2.set_xlabel('Improvement (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Relative Improvement', fontsize=16, fontweight='bold', pad=15)
    ax2.axvline(x=0, color='black', linewidth=1, linestyle='-', alpha=0.3)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        ax2.text(val + (1 if val > 0 else -1), i,
                f'{val:+.1f}%',
                ha='left' if val > 0 else 'right',
                va='center', fontsize=11, fontweight='bold')
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Scatter baseline
    ax3 = fig.add_subplot(gs[1, 0])
    
    y_true_baseline = test_results_baseline['temp'].values
    y_pred_baseline = test_results_baseline['predicted'].values
    
    ax3.scatter(y_true_baseline, y_pred_baseline, alpha=0.6, s=50,
                c='#95a5a6', edgecolors='white', linewidth=0.5)
    
    # line
    min_val = min(y_true_baseline.min(), y_pred_baseline.min())
    max_val = max(y_true_baseline.max(), y_pred_baseline.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5)
    
    ax3.set_xlabel('Observed Temperature (°C)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Predicted Temperature (°C)', fontsize=12, fontweight='bold')
    ax3.set_title('Simple Kriging: Predicted vs Observed', fontsize=14, fontweight='bold')
    ax3.grid(alpha=0.3, linestyle='--')
    
    # Add R² text
    r2_baseline = comparison_df.loc['R²', comparison_df.columns[0]]
    ax3.text(0.05, 0.95, f'R² = {r2_baseline:.3f}',
            transform=ax3.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Scatter hybrid
    ax4 = fig.add_subplot(gs[1, 1])
    
    y_true_hybrid = test_results_hybrid['temp'].values
    y_pred_hybrid = test_results_hybrid['predicted'].values
    
    ax4.scatter(y_true_hybrid, y_pred_hybrid, alpha=0.6, s=50,
                c='#3498db', edgecolors='white', linewidth=0.5)
    
    ax4.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5)
    
    ax4.set_xlabel('Observed Temperature (°C)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Predicted Temperature (°C)', fontsize=12, fontweight='bold')
    ax4.set_title('HRMTA: Predicted vs Observed', fontsize=14, fontweight='bold')
    ax4.grid(alpha=0.3, linestyle='--')
    
    r2_hybrid = comparison_df.loc['R²', comparison_df.columns[1]]
    ax4.text(0.05, 0.95, f'R² = {r2_hybrid:.3f}',
            transform=ax4.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Error distributions
    ax5 = fig.add_subplot(gs[1, 2])
    
    errors_baseline = y_pred_baseline - y_true_baseline
    errors_hybrid = y_pred_hybrid - y_true_hybrid
    
    ax5.hist(errors_baseline, bins=30, alpha=0.6, label='Simple Kriging',
            color='#95a5a6', edgecolor='white', linewidth=1)
    ax5.hist(errors_hybrid, bins=30, alpha=0.6, label='HRMTA',
            color='#3498db', edgecolor='white', linewidth=1)
    
    ax5.axvline(x=0, color='black', linewidth=2, linestyle='--', alpha=0.5)
    ax5.set_xlabel('Prediction Error (°C)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax5.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=11)
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    print(f"\n✓ Model comparison plot saved to: {output_path}")
    plt.close()

def create_comparison_summary_image(
    comparison_df: pd.DataFrame,
    output_path: Path = None
):
    """
    Create a simple, clean comparison summary image (for presentations).
    """
    if output_path is None:
        from .config import RUN_OUTPUT_DIR
        output_path = RUN_OUTPUT_DIR / "comparison_summary.png"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    
    # Title
    fig.text(0.5, 0.95, 'Model Performance Comparison', 
            fontsize=20, fontweight='bold', ha='center', color='#2c3e50')
    
    # Subtitle
    fig.text(0.5, 0.88, 'Simple Kriging vs HRMTA',
            fontsize=14, ha='center', color='#7f8c8d', style='italic')
    
    # Create table
    metrics = ['RMSE (°C)', 'MAE (°C)', 'R²', 'Bias (°C)']
    baseline_vals = [
        comparison_df.loc['RMSE', comparison_df.columns[0]],
        comparison_df.loc['MAE', comparison_df.columns[0]],
        comparison_df.loc['R²', comparison_df.columns[0]],
        comparison_df.loc['Bias', comparison_df.columns[0]]
    ]
    hybrid_vals = [
        comparison_df.loc['RMSE', comparison_df.columns[1]],
        comparison_df.loc['MAE', comparison_df.columns[1]],
        comparison_df.loc['R²', comparison_df.columns[1]],
        comparison_df.loc['Bias', comparison_df.columns[1]]
    ]
    improvements = [
        comparison_df.loc['RMSE', '% Improvement'],
        comparison_df.loc['MAE', '% Improvement'],
        comparison_df.loc['R²', '% Improvement'],
        comparison_df.loc['Bias', '% Improvement']
    ]
    
    # Table data
    table_data = []
    for i, metric in enumerate(metrics):
        table_data.append([
            metric,
            f"{baseline_vals[i]:.3f}",
            f"{hybrid_vals[i]:.3f}",
            f"{improvements[i]:+.1f}%"
        ])
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=['Metric', 'Simple Kriging', 'HRMTA', 'Improvement'],
        cellLoc='center',
        loc='center',
        bbox=[0.1, 0.2, 0.8, 0.6]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style cells
    for i in range(1, 5):
        for j in range(4):
            if j == 3:  # improvement column
                val = float(table_data[i-1][3].rstrip('%'))
                if val > 0:
                    table[(i, j)].set_facecolor('#d5f4e6')
                else:
                    table[(i, j)].set_facecolor('#ffe6e6')
            elif j == 2:  # hybrid column
                table[(i, j)].set_facecolor('#e8f4f8')
    
    ax.axis('off')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Comparison summary saved to: {output_path}")
    plt.close()
