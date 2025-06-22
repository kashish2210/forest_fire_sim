import numpy as np
import rasterio
from forestfire_sim.config import DEM, PREDICTION_MAP
import matplotlib.pyplot as plt
import imageio
import os

def simulate_fire_spread(prob_map_path, dem_path, out_dir="outputs/", hours=[1, 2, 3, 6, 12]):
    os.makedirs(out_dir, exist_ok=True)

    # Load fire probability map
    with rasterio.open(prob_map_path) as src:
        fire_map = src.read(1)
        profile = src.profile

    # Load elevation for slope dummy (here used as static terrain factor)
    with rasterio.open(dem_path) as dem_src:
        slope = dem_src.read(1)

    # Create binary ignition zones
    ignition = (fire_map > 0.48).astype(np.uint8)#temp treshold for ignition

    # Create empty spread map
    H, W = ignition.shape
    spread = np.zeros((H, W), dtype=np.uint8)
    spread[ignition == 1] = 1

    # Store frames for animation
    frames = [spread.copy()]

    # Dummy rule: spread to 8-neighbors
    for hour in range(1, max(hours) + 1):
        new_spread = spread.copy()

        for i in range(1, H-1):
            for j in range(1, W-1):
                if spread[i, j] == 1:
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if (dx, dy) != (0, 0):
                                ni, nj = i + dx, j + dy
                                # Random + terrain filter
                                if np.random.rand() > 0.4 and slope[ni, nj] < 0.8:
                                    new_spread[ni, nj] = 1

        spread = new_spread.copy()

        if hour in hours:
            # Save raster at milestone hour
            hour_path = os.path.join(out_dir, f"spread_{hour}h.tif")
            with rasterio.open(hour_path, "w", **profile) as dst:
                dst.write(spread.astype("uint8"), 1)
            print(f"Saved: {hour_path}")

        # Save frame for animation
        frames.append(spread.copy())

    # Save animation
    gif_path = os.path.join(out_dir, "spread_1_2_3_6_12h.gif")
    imageio.mimsave(gif_path, [frame * 255 for frame in frames], duration=0.5)
    print(f"Fire spread animation saved at: {gif_path}")
