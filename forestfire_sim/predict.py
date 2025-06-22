import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio

# If running in notebook: manually set paths (optional fallback)
try:
    from config import DEM, FEATURE_STACK, PREDICTION_MAP
except ImportError:
    DEM = "data/uttarakhand_dem.tif"
    FEATURE_STACK = "data/feature_stack.npz"
    PREDICTION_MAP = "outputs/prediction_day1.tif"

# -----------------------------
# 🔶 U-Net Model Definition
# -----------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=7, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_c)
            )

        self.enc1 = nn.Sequential(conv_block(in_channels, 64), conv_block(64, 64))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(conv_block(64, 128), conv_block(128, 128))
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(conv_block(128, 256), conv_block(256, 256))

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(conv_block(256, 128), conv_block(128, 64))
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.final = nn.Conv2d(128, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        out = self.final(d1)
        return self.sigmoid(out)

# -----------------------------
# 🔷 Dataset
# -----------------------------
class FireDataset(Dataset):
    def __init__(self, feature_path, label_path, patch_size=64):
        self.data = np.load(feature_path)["features"]  # (C, H, W)
        self.labels = np.load(label_path)             # (H, W)
        self.patch_size = patch_size

        self.indices = []
        h, w = self.labels.shape
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                self.indices.append((i, j))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i, j = self.indices[idx]
        x = self.data[:, i:i+self.patch_size, j:j+self.patch_size]
        y = self.labels[i:i+self.patch_size, j:j+self.patch_size]
        return torch.FloatTensor(x), torch.FloatTensor(y).unsqueeze(0)

# -----------------------------
# 🧪 Train U-Net
# -----------------------------
def train_model():
    dataset = FireDataset("data/feature_stack.npz", "data/target_labels.npy", patch_size=64)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = UNet(in_channels=dataset.data.shape[0])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("🧠 Training U-Net on patch dataset...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/unet_fire_model.pth")
    print("✅ Model saved to models/unet_fire_model.pth")

# -----------------------------
# 🔮 Predict Full Image
# -----------------------------
def predict_full_image():
    print("🛰  Predicting full-region fire probability map...")

    data = np.load(FEATURE_STACK)["features"]  # (C, H, W)
    c, h, w = data.shape
    model = UNet(in_channels=c)
    model.load_state_dict(torch.load("models/unet_fire_model.pth", map_location="cpu"))
    model.eval()

    input_tensor = torch.FloatTensor(data).unsqueeze(0)  # (1, C, H, W)
    with torch.no_grad():
        output = model(input_tensor)[0, 0].numpy()  # (H, W)

    fire_prob_map = np.clip(output, 0, 1)

    # Save as GeoTIFF
    with rasterio.open(DEM) as src:
        profile = src.profile
        profile.update(count=1, dtype="float32")

        os.makedirs(os.path.dirname(PREDICTION_MAP), exist_ok=True)
        with rasterio.open(PREDICTION_MAP, "w", **profile) as dst:
            dst.write(fire_prob_map.astype("float32"), 1)

    print("🔥 Fire probability saved to:", PREDICTION_MAP)

# -----------------------------
# ▶️ Main Runner
# -----------------------------
if __name__ == "__main__":
    train_model()
    predict_full_image()
