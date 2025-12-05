"""
Export utilities for GIS integration.
"""
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pathlib import Path
from datetime import datetime

from .config import OUTPUT_DIR, CRS_POLAND, CRS_WGS84
from .utils import PL_GEOMETRY_2180

def export_grid_to_geotiff(
    grid_data: np.ndarray,
    grid_x_1d: np.ndarray,
    grid_y_1d: np.ndarray,
    output_path: Path,
    variable_name: str = "temperature",
    units: str = "degrees_celsius",
    crs: str = CRS_POLAND,
    metadata: dict = None
):
    """
    Export grid data to GeoTIFF
    
    Args:
        grid_data: 2D numpy array with data
        grid_x_1d, grid_y_1d: 1D coordinate arrays
        output_path: Output file path
        variable_name: Name of the variable
        units: Units of the data
        crs: Coordinate reference system
        metadata: Additional metadata dictionary
    """
    print(f"\nExporting to GeoTIFF: {output_path.name}")
    
    # Get grid bounds
    bounds = PL_GEOMETRY_2180.bounds
    
    # Create affine transform
    transform = from_bounds(
        bounds[0], bounds[1], bounds[2], bounds[3],
        len(grid_x_1d), len(grid_y_1d)
    )
    
    # Handle NaN values, convert to nodata value
    nodata_value = -9999.0
    data_export = grid_data.copy()
    data_export[np.isnan(grid_data)] = nodata_value
    
    # Prepare metadata
    meta = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': nodata_value,
        'width': len(grid_x_1d),
        'height': len(grid_y_1d),
        'count': 1,
        'crs': CRS.from_string(crs),
        'transform': transform,
        'compress': 'lzw',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256
    }
    
    # Write GeoTIFF
    with rasterio.open(output_path, 'w', **meta) as dst:
        # Write data
        dst.write(data_export.astype('float32'), 1)
        
        # Set band description
        dst.set_band_description(1, variable_name)
        
        # Add custom metadata
        tags = {
            'VARIABLE_NAME': variable_name,
            'UNITS': units,
            'CREATION_DATE': datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            'MODEL_VERSION': 'HRMTA v1.0',
            'RESOLUTION': f'{abs(transform[0]):.1f} meters',
            'MIN_VALUE': f'{np.nanmin(grid_data):.2f}',
            'MAX_VALUE': f'{np.nanmax(grid_data):.2f}',
            'MEAN_VALUE': f'{np.nanmean(grid_data):.2f}',
            'VALID_PIXELS': f'{np.sum(~np.isnan(grid_data))}',
        }
        
        # add custom metadata if provided
        if metadata:
            tags.update(metadata)
        
        dst.update_tags(**tags)
    
    # Get file size
    file_size = output_path.stat().st_size / (1024 * 1024)  # in MB
    
    print(f"✓ GeoTIFF exported successfully")
    print(f"  File size: {file_size:.2f} MB")
    print(f"  Dimensions: {len(grid_y_1d)} x {len(grid_x_1d)}")
    print(f"  CRS: {crs}")
    print(f"  Bounds: {bounds}")
    print(f"  Data range: {np.nanmin(grid_data):.2f} to {np.nanmax(grid_data):.2f} {units}")

def export_temperature_products(
    temperature_grid: np.ndarray,
    uncertainty_grid: np.ndarray,
    grid_x_1d: np.ndarray,
    grid_y_1d: np.ndarray,
    metrics: dict = None,
    export_wgs84: bool = True
):
    """
    Export all temperature products (temperature, uncertainty, etc.) to GeoTIFF
    
    Args:
        temperature_grid: Temperature data (2D array)
        uncertainty_grid: Uncertainty data (2D array, optional)
        grid_x_1d, grid_y_1d: Coordinate arrays
        metrics: Model performance metrics
        export_wgs84: Also export in WGS84 for web mapping
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Prepare metadata
    metadata = {
        'DESCRIPTION': 'High-Resolution Mesoscale Temperature Analysis',
        'SOURCE': 'IMGW, Netatmo, TraxElektronik',
        'METHOD': 'Physics-Informed Hybrid (Lapse Rate + LightGBM + Kriging)',
    }
    
    if metrics:
        metadata.update({
            'TEST_RMSE': f"{metrics.get('RMSE', 0):.3f} °C",
            'TEST_R2': f"{metrics.get('R²', 0):.3f}",
            'TEST_MAE': f"{metrics.get('MAE', 0):.3f} °C",
        })
    
    # Export temperature in EPSG:2180 (Poland)
    temp_file = OUTPUT_DIR / f"temperature_2180_{timestamp}.tif"
    export_grid_to_geotiff(
        temperature_grid,
        grid_x_1d,
        grid_y_1d,
        temp_file,
        variable_name="air_temperature",
        units="degrees_celsius",
        crs=CRS_POLAND,
        metadata=metadata
    )
    
    # Export uncertainty if available
    if uncertainty_grid is not None:
        unc_file = OUTPUT_DIR / f"uncertainty_2180_{timestamp}.tif"
        export_grid_to_geotiff(
            uncertainty_grid,
            grid_x_1d,
            grid_y_1d,
            unc_file,
            variable_name="prediction_uncertainty",
            units="degrees_celsius",
            crs=CRS_POLAND,
            metadata={**metadata, 'DESCRIPTION': 'Ensemble prediction uncertainty (std dev)'}
        )
    
    # Export in WGS84 for web mapping (optional)
    if export_wgs84:
        print("\nReprojecting to EPSG:4326 for web compatibility...")
        
        # Temperature in WGS84
        temp_wgs84_file = OUTPUT_DIR / f"temperature_wgs84_{timestamp}.tif"
        reproject_to_wgs84(temp_file, temp_wgs84_file)
        
        # Uncertainty in WGS84
        if uncertainty_grid is not None:
            unc_wgs84_file = OUTPUT_DIR / f"uncertainty_wgs84_{timestamp}.tif"
            reproject_to_wgs84(unc_file, unc_wgs84_file)
    
    return temp_file

def reproject_to_wgs84(input_path: Path, output_path: Path):
    """
    Reproject GeoTIFF from EPSG:2180 to WGS84
    
    Args:
        input_path: Input GeoTIFF in EPSG:2180
        output_path: Output GeoTIFF in WGS84
    """
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    
    with rasterio.open(input_path) as src:
        # Calculate transform for WGS84
        transform, width, height = calculate_default_transform(
            src.crs,
            CRS.from_string(CRS_WGS84),
            src.width,
            src.height,
            *src.bounds
        )
        
        # Update metadata
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': CRS.from_string(CRS_WGS84),
            'transform': transform,
            'width': width,
            'height': height
        })
        
        # Reproject
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=CRS.from_string(CRS_WGS84),
                    resampling=Resampling.bilinear
                )
            
            # Copy metadata
            dst.update_tags(**src.tags())
    
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ WGS84 version created: {output_path.name} ({file_size:.2f} MB)")