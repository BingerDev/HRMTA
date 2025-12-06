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
    GRID_RESOLUTION, DISPLAY_STATION_SOURCES
)
from .utils import PL_BOUNDARY_WGS84

def load_color_scale_advanced():
    """
    Load color scale as a discrete ListedColormap with BoundaryNorm.
    """
    if not COLOR_SCALE.exists():
        print(f"⚠️  Color scale not found: {COLOR_SCALE}")
        # Fallback to standard matplotlib
        cmap = plt.cm.RdYlBu_r
        norm = Normalize(vmin=-40, vmax=40)
        return cmap, norm, -40, 40
    
    try:
        df = pd.read_csv(COLOR_SCALE).sort_values("value")
        
        if 'value' in df.columns and 'color' in df.columns:
            # Extract bounds and hex colors
            bounds = df['value'].to_numpy()
            colors_hex = df['color'].values
            
            # Convert to RGB list
            colors_rgb = [mcolors.to_rgb(c) for c in colors_hex]
            
            # Create Discrete Colormap and BoundaryNorm
            # note: listed colormap needs N colors, boundaries need N+1 points usually,
            # but BoundaryNorm with clip=True handles mapping values to specific bins.
            cmap = ListedColormap(colors_rgb, name="csv_scale")
            norm = BoundaryNorm(bounds, cmap.N, clip=True)
            
            vmin = bounds.min()
            vmax = bounds.max()
            
            print(f"✓ Loaded discrete color scale: {vmin}°C to {vmax}°C")
            return cmap, norm, vmin, vmax
        else:
            print("⚠️  Invalid color scale format")
            return plt.cm.RdYlBu_r, Normalize(vmin=-40, vmax=40), -40, 40
            
    except Exception as e:
        print(f"⚠️  Error loading color scale: {e}")
        return plt.cm.RdYlBu_r, Normalize(vmin=-40, vmax=40), -40, 40

def plot_temperature_map(
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    temperature: np.ndarray,
    stations_gdf: gpd.GeoDataFrame = None,
    output_path: str = None,
    show: bool = True,
    title_suffix: str = "",
    resolution_km=GRID_RESOLUTION
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

    # Load Data & Scale
    cmap, norm, scale_vmin, scale_vmax = load_color_scale_advanced()
    
    # Setup Figure Geometry based on Poland's aspect ratio
    # calculate aspect ratio correction: aspect_deg * cos(mid_lat)
    pl_bounds = PL_BOUNDARY_WGS84.total_bounds # [minx, miny, maxx, maxy]
    minx, miny, maxx, maxy = pl_bounds
    
    lat_mid = (miny + maxy) / 2
    aspect_true = (maxx - minx) * cos(radians(lat_mid)) / (maxy - miny)
    
    FIG_W = 13  # Fixed width from reference
    FIG_H = FIG_W / aspect_true
    
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    
    # Plot Raster
    # using pcolormesh to support non-rectilinear grids if necessary 
    img = ax.pcolormesh(
        grid_lon, grid_lat, temperature,
        cmap=cmap,
        norm=norm,
        shading='nearest', # Equivalent to interpolation='nearest'
        rasterized=True,
        zorder=1
    )
    
    # Borders & Neatline
    # Inner borders
    PL_BOUNDARY_WGS84.boundary.plot(
        ax=ax,
        edgecolor="#333333",
        linewidth=1.0,
        zorder=2
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

        # Markers
        ax.scatter(
            stations_to_plot.geometry.x, 
            stations_to_plot.geometry.y,
            s=20, marker="s", color="black",
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
        
        # Iterate and plot text with transform
        for idx, row in stations_to_plot.iterrows():
            if pd.isna(row[temp_col]): continue
            
            label_text = f"{row[temp_col]:.1f}"
            txt = ax.text(row.geometry.x, row.geometry.y, label_text, **txt_kw)
            
            # Apply offset transform
            txt.set_transform(txt.get_transform() + 
                              mtransforms.ScaledTranslation(dx_pt/72, 0, fig.dpi_scale_trans))

    # Titles & Footer
    utc_now = datetime.utcnow().strftime('%Y-%m-%d  %H:%M  UTC')
    
    # Title aligned left
    resolution_km_display = resolution_km / 1000
    ax.set_title(
        f'Polska • Temperatura powietrza 2 m • {resolution_km_display:g} km\n{utc_now}',
        loc='left', pad=10, fontsize=12, weight='bold'
    )
    
    # Source footer
    fig.text(
        0.02, 0.01,
        'Źródła pochodzenia danych obserwacyjnych: IMGW (Instytut Meteorologii i Gospodarki Wodnej)  •  TraxElektronik  •  Netatmo',
        fontsize=6, color='#444444'
    )
    
    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.12)
    
    cb = fig.colorbar(img, cax=cax)
    
    # Ticks every 5 degrees based on scale range
    ticks5 = np.arange(
        np.floor(scale_vmin/5)*5,
        np.ceil(scale_vmax/5)*5 + 1, 
        5, dtype=int
    )
    cb.set_ticks(ticks5)
    cb.set_ticklabels(ticks5)
    cb.ax.tick_params(labelsize=9)
    cb.set_label('Temperatura 2 m (°C)', rotation=270, labelpad=15, fontsize=9)
    
    # Final settings
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect('auto') # handled aspect in figsize
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Map saved to: {output_path}")
    
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
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_position([0.02, 0.03, 0.80, 0.85])
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
    
    # Poland boundary
    PL_BOUNDARY_WGS84.boundary.plot(ax=ax, edgecolor='#2c3e50', linewidth=2.5)
    
    pl_bounds = PL_BOUNDARY_WGS84.total_bounds
    ax.set_xlim(pl_bounds[0], pl_bounds[2])
    ax.set_ylim(pl_bounds[1], pl_bounds[3])

    center_lat = (pl_bounds[1] + pl_bounds[3]) / 2
    ax.set_aspect(1 / np.cos(np.radians(center_lat)), adjustable='box')
    ax.axis('off')
    
    # Title
    fig.text(0.5, 0.96, title, fontsize=32, fontweight='bold', 
             ha='center', va='top', color='#2c3e50')
    
    subtitle = f"Mean: {np.nanmean(uncertainty):.3f}°C, 95th percentile: {np.nanpercentile(uncertainty, 95):.3f}°C"
    fig.text(0.5, 0.91, subtitle, fontsize=14, ha='center', va='top', 
            color='#7f8c8d', style='italic')
    
    # Colorbar
    cax = fig.add_axes([0.85, 0.10, 0.025, 0.75])
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label('Uncertainty (°C)', fontsize=15, color='#2c3e50', weight='bold')
    cbar.ax.tick_params(labelsize=13, colors='#2c3e50')
    cbar.outline.set_visible(False)
    
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='#f8f9fa')
    print(f"✓ Uncertainty map saved to: {output_path}")

def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20):
    """Modern feature importance plot."""
    from .config import OUTPUT_DIR
    
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
    plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=200, 
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
        from .config import OUTPUT_DIR
        output_path = OUTPUT_DIR / "model_comparison.png"
    
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
        from .config import OUTPUT_DIR
        output_path = OUTPUT_DIR / "comparison_summary.png"
    
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
