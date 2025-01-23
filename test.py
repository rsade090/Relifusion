import torch
from torch.utils.data import DataLoader
from nuscenes_loader import NuScenesDataset
from relifusion_model import ReliFusion
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Argument Parser
parser = argparse.ArgumentParser(description="Test ReliFusion Model on NuScenes Dataset")
parser.add_argument('--nuscenes_root', type=str, required=True, help="Path to the NuScenes dataset root directory")
parser.add_argument('--version', type=str, default="v1.0-mini", help="NuScenes dataset version (default: v1.0-mini)")
parser.add_argument('--batch_size', type=int, default=2, help="Batch size for DataLoader (default: 2)")
parser.add_argument('--num_workers', type=int, default=4, help="Number of DataLoader workers (default: 4)")
parser.add_argument('--device', type=str, default="cuda", help="Device to run the model on (default: cuda)")
parser.add_argument('--split', type=str, default="val", help="Dataset split to use (default: val)")
args = parser.parse_args()

# Configuration from Arguments
NUSCENES_ROOT = args.nuscenes_root
VERSION = args.version
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
DEVICE = args.device
SPLIT = args.split

# Initialize Dataset and DataLoader
print("Loading NuScenes Dataset...")
dataset = NuScenesDataset(root_dir=NUSCENES_ROOT, version=VERSION, split=SPLIT, t=1)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Initialize Model
print("Initializing ReliFusion Model...")
model = ReliFusion(lidar_input_dim=4, camera_input_channels=3, hidden_dim=256).to(DEVICE)
model.eval()

# Visualization Function
def visualize_output(lidar_data, camera_data, outputs):
    """Visualize model inputs and outputs."""
    batch_size = lidar_data.size(0)
    for i in range(batch_size):
        plt.figure(figsize=(12, 6))

        # Visualize Camera Input
        plt.subplot(1, 3, 1)
        plt.title("Camera Input")
        cam_image = camera_data[i].cpu().numpy().transpose(1, 2, 0)  # Convert to HxWxC
        plt.imshow(cam_image / np.max(cam_image))
        plt.axis("off")

        # Visualize LiDAR Input
        plt.subplot(1, 3, 2)
        plt.title("LiDAR Input (Projection)")
        lidar_points = lidar_data[i][0].cpu().numpy()  # Taking first channel as example
        plt.imshow(lidar_points, cmap="viridis")
        plt.axis("off")

        # Visualize Model Output
        plt.subplot(1, 3, 3)
        plt.title("Model Output")
        output_features = outputs[i].cpu().numpy()
        plt.imshow(output_features, cmap="coolwarm")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

# Test Loop
print("Starting Inference...")
with torch.no_grad():
    for batch_idx, data in enumerate(dataloader):
        # Extract LiDAR and Camera Data
        lidar_data = data['lidar'].to(DEVICE)  # LiDAR input: (B, 4, D, H, W)
        camera_data = torch.stack([torch.load(img).to(DEVICE) for img in data['camera']['CAM_FRONT']])  # Example for CAM_FRONT

        # Forward Pass
        outputs = model(lidar_data, camera_data)

        # Visualize Results
        visualize_output(lidar_data, camera_data, outputs)

        # Process Results
        print(f"Batch {batch_idx + 1}/{len(dataloader)}: Output Shape: {outputs.shape}")

print("Inference Completed.")
