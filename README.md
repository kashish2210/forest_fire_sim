# Forest Fire Simulation in Uttarakhand

A deep learning pipeline for **forest fire prediction and spread simulation** using geospatial and weather data for the Uttarakhand region.

```yaml
## Project Structure

forestfire_sim/
├── data/
│ ├── uttarakhand_dem.tif # Elevation data
│ ├── lulc_uttarakhand.tif # Land Use / Land Cover data
│ ├── weather_2024.nc # Weather data (NetCDF)
│ ├── target_labels.npy # Fire labels (VIIRS-style)
│
├── outputs/
│ ├── prediction_day1.tif # Model-predicted fire probability
│ ├── spread_1_2_3_6_12h.gif # Animated fire spread
│
├── forestfire_sim/
│ ├── init.py
│ ├── config.py # Paths and config constants
│ ├── preprocess.py # Data preprocessing & normalization
│ ├── model.py # U-Net fire prediction model
│ ├── train.py # Model training script
│ ├── predict.py # Predict fire probability
│ ├── simulate.py # Fire spread simulation
│ ├── visualize.py # Visualization tools
```

## ⚙️ Installation

### 📌 Requirements
- Python 3.10+
- Conda (recommended)

### 🐍 Create Environment

```bash
conda create -n forestfire python=3.10
conda activate forestfire
```
# Install dependencies
```
pip install -r requirements.txt
```
Or, if using Conda YAML:
```
conda env create -f environment.yml
conda activate forestfire
```
# How to Use
1. Preprocess Input Data
```
python -m forestfire_sim.preprocess
```
This will:

Normalize and stack DEM, LULC, and weather layers.

Generate dummy binary fire labels from a VIIRS-style raster.

2. Train the Model
```
python -m forestfire_sim.train
```
Trains a simple U-Net to predict fire zones from raster inputs.

3. 📈 Predict Fire Probability
```
python -m forestfire_sim.predict
```
Generates:

outputs/prediction_day1.tif: predicted fire probabilities per pixel.

4. 🔍 Visualize Results
```
python -m forestfire_sim.visualize
```
Includes:

Heatmaps of fire probability.

Overlay of fire zones on DEM.

Histogram of predicted probabilities.

5. Simulate Fire Spread
```
python -m forestfire_sim.simulate
```
Creates:
```
outputs/spread_1_2_3_6_12h.gif: Fire spread simulation over time.
```

📊 Sample Output
Fire prediction heatmap

Fire zones mask (thresholded)

Fire spread animation


🧪 Technologies Used
rasterio for geospatial data

xarray for NetCDF handling

PyTorch for deep learning

matplotlib for visualization

imageio for simulation GIFs

📌 Notes
Paths and thresholds are set in forestfire_sim/config.py.

The dataset currently uses dummy/mock data for demonstration.

Model is basic; replace with custom architecture for production.

🤝 Contributing
Feel free to fork this repo and open issues or PRs.

