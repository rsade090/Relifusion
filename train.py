import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nuscenes_loader import NuScenesDataset
from relifusion_model import ReliFusion
from logger import Logger
from torch.utils.tensorboard import SummaryWriter
import argparse

# Argument Parser
parser = argparse.ArgumentParser(description="Train ReliFusion Model on NuScenes Dataset")
parser.add_argument('--nuscenes_root', type=str, required=True, help="Path to the NuScenes dataset root directory")
parser.add_argument('--version', type=str, default="v1.0-mini", help="NuScenes dataset version (default: v1.0-mini)")
parser.add_argument('--batch_size', type=int, default=4, help="Batch size for DataLoader (default: 4)")
parser.add_argument('--num_workers', type=int, default=4, help="Number of DataLoader workers (default: 4)")
parser.add_argument('--device', type=str, default="cuda", help="Device to train the model on (default: cuda)")
parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs (default: 10)")
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for optimizer (default: 1e-4)")
parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay for optimizer (default: 1e-5)")
parser.add_argument('--save_path', type=str, default="model.pth", help="Path to save the trained model (default: model.pth)")
parser.add_argument('--log_dir', type=str, default="logs", help="Directory to save training logs (default: logs)")
args = parser.parse_args()

# Configuration from Arguments
NUSCENES_ROOT = args.nuscenes_root
VERSION = args.version
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
DEVICE = args.device
EPOCHS = args.epochs
LR = args.lr
WEIGHT_DECAY = args.weight_decay
SAVE_PATH = args.save_path
LOG_DIR = args.log_dir

# Initialize Logger and TensorBoard Writer
logger = Logger(log_dir=LOG_DIR, log_file="training.log")
writer = SummaryWriter(log_dir=LOG_DIR)

# Initialize Dataset and DataLoader
logger.info("Loading NuScenes Dataset...")
train_dataset = NuScenesDataset(root_dir=NUSCENES_ROOT, version=VERSION, split="train", t=5)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

# Initialize Model
logger.info("Initializing ReliFusion Model...")
model = ReliFusion(lidar_input_dim=4, camera_input_channels=3, hidden_dim=256).to(DEVICE)

# Loss and Optimizer
logger.info("Setting up Loss and Optimizer...")
criterion = nn.MSELoss()  # Example: replace with appropriate multi-task loss
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Training Loop
logger.info("Starting Training...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for batch_idx, data in enumerate(train_loader):
        # Extract LiDAR and Camera Data
        lidar_data = data['lidar'].to(DEVICE)  # LiDAR input: (B, 4, D, H, W)
        camera_data = torch.stack([torch.load(img).to(DEVICE) for img in data['camera']['CAM_FRONT']])  # Example for CAM_FRONT

        # Forward Pass
        optimizer.zero_grad()
        outputs = model(lidar_data, camera_data)

        # Compute Loss
        # Example loss: compute based on ground truth (replace with actual task-specific labels)
        labels = torch.zeros_like(outputs).to(DEVICE)  # Placeholder labels
        loss = criterion(outputs, labels)

        # Backward Pass and Optimization
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        logger.info(f"Epoch [{epoch + 1}/{EPOCHS}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Log to TensorBoard
        writer.add_scalar("Loss/Batch", loss.item(), epoch * len(train_loader) + batch_idx)

    avg_loss = epoch_loss / len(train_loader)
    logger.info(f"Epoch [{epoch + 1}/{EPOCHS}] completed with Average Loss: {avg_loss:.4f}")
    writer.add_scalar("Loss/Epoch", avg_loss, epoch)

# Save Model
logger.info(f"Saving Model to {SAVE_PATH}...")
torch.save(model.state_dict(), SAVE_PATH)
logger.info("Training Completed and Model Saved.")
writer.close()
