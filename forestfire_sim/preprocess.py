import rasterio
import numpy as np
import xarray as xr
import rioxarray as rxr
from scipy.ndimage import sobel
import os


def read_raster(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1)
        profile = src.profile
    return data, profile


def compute_slope_aspect(dem_array):
    dx = sobel(dem_array, axis=1)
    dy = sobel(dem_array, axis=0)

    slope = np.sqrt(dx**2 + dy**2)
    slope_deg = np.arctan(slope) * (180 / np.pi)

    aspect = np.arctan2(-dx, dy)
    aspect_deg = (aspect * 180 / np.pi) % 360

    return slope_deg, aspect_deg


def read_weather_netcdf(file_path, variable):
    ds = xr.open_dataset(file_path)
    data_slice = ds[variable].isel(time=0)
    return data_slice.values


def stack_features(dem_path, lulc_path, weather_nc_path):
    dem, profile = read_raster(dem_path)
    lulc, _ = read_raster(lulc_path)

    slope, aspect = compute_slope_aspect(dem)

    temp = read_weather_netcdf(weather_nc_path, "temperature")
    humidity = read_weather_netcdf(weather_nc_path, "humidity")
    wind = read_weather_netcdf(weather_nc_path, "wind_speed")

    stacked = np.stack([dem, slope, aspect, lulc, temp, humidity, wind], axis=0)
    return stacked, profile


def normalize_channels(data_stack):
    norm_stack = np.zeros_like(data_stack, dtype=np.float32)
    for i in range(data_stack.shape[0]):
        band = data_stack[i]
        min_val = np.nanmin(band)
        max_val = np.nanmax(band)
        norm_stack[i] = (band - min_val) / (max_val - min_val + 1e-6)
    return norm_stack


def save_stack_as_npz(stack, save_path="data/feature_stack.npz"):
    np.savez_compressed(save_path, features=stack)
    print(f"Feature stack saved to: {save_path}")


def prepare_target_label(viirs_path, target_path, threshold=0):
    viirs_data, _ = read_raster(viirs_path)
    label = (viirs_data > threshold).astype(np.uint8)
    np.save(target_path, label)
    print(f"Saved binary label map to: {target_path}")
    return label

if __name__ == "__main__":
    # Paths
    DEM = "data/uttarakhand_dem.tif"
    LULC = "data/lulc_uttarakhand.tif"
    WEATHER = "data/weather_2024.nc"
    VIIRS = "data/viirs_fire_2023.tif"

    print("📥 Reading and stacking features...")
    features, meta = stack_features(DEM, LULC, WEATHER)
    print("✅ Original shape:", features.shape)

    print("⚖️ Normalizing feature channels...")
    norm_features = normalize_channels(features)
    print("✅ Normalized shape:", norm_features.shape)

    print("💾 Saving feature stack...")
    save_stack_as_npz(norm_features)

    print("🔥 Preparing target label map...")
    prepare_target_label(VIIRS, "data/target_labels.npy")
