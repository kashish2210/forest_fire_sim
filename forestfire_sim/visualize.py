import numpy as np
import matplotlib.pyplot as plt
import rasterio
import os

def show_fire_probability_map(prediction_path, threshold=0.6):
    """
    Visualize the fire probability map and highlight high-risk zones.
    """
    with rasterio.open(prediction_path) as src:
        fire_map = src.read(1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot full fire probability map
    im1 = axs[0].imshow(fire_map, cmap="hot", vmin=0, vmax=1)
    axs[0].set_title("Fire Probability Map")
    fig.colorbar(im1, ax=axs[0], label="Probability")

    # Plot binary fire zones above threshold
    fire_zones = (fire_map > threshold).astype(np.uint8)
    im2 = axs[1].imshow(fire_zones, cmap="Reds")
    axs[1].set_title(f"Fire Zones (prob > {threshold})")
    fig.tight_layout()
    plt.show()

def show_label_overlay(dem_path, label_path):
    """
    Overlay binary fire label map on DEM for spatial context.
    """
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
    
    label = np.load(label_path)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(dem, cmap="Greys", interpolation="none")
    ax.imshow(label, cmap="Reds", alpha=0.5)
    ax.set_title("Fire Labels Overlay on DEM")
    plt.colorbar(ax.imshow(dem, cmap="pink"), label="Elevation")
    plt.axis("off")
    plt.show()

def preview_fire_seed(prob_map_path, threshold=0.6):
    """
    Visualize the seed zones selected for fire spread simulation.
    """
    with rasterio.open(prob_map_path) as src:
        prob_map = src.read(1)

    initial_fire = (prob_map > threshold).astype(np.uint8)

    plt.imshow(initial_fire, cmap="hot")
    plt.title("Initial Fire Seeds (from threshold)")
    plt.colorbar(label="Fire Seed (1=Fire)")
    plt.axis("off")
    plt.show()

def summarize_prediction_stats(prediction_path):
    """
    Print basic statistics from predicted fire probability map.
    """
    with rasterio.open(prediction_path) as src:
        fire_map = src.read(1)
        print("Fire Prediction Stats")
        print("Min:", np.min(fire_map))
        print("Max:", np.max(fire_map))
        print("Mean:", np.mean(fire_map))

# Example usage block
# if __name__ == "__main__":
#     DEM_PATH = "data/uttarakhand_dem.tif"
#     LABEL_PATH = "data/target_labels.npy"
#     PRED_PATH = "outputs/prediction_day1.tif"

#     show_label_overlay(DEM_PATH, LABEL_PATH)
#     show_fire_probability_map(PRED_PATH, threshold=0.52)
#     preview_fire_seed(PRED_PATH, threshold=0.52)
#     summarize_prediction_stats(PRED_PATH)

def simulate():
    